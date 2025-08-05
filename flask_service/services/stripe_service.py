import stripe
import logging
from flask import current_app, session
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def initialize_stripe():
    """Initialize Stripe with secret key."""
    stripe.api_key = current_app.config['STRIPE_SECRET_KEY']
    if not stripe.api_key:
        raise ValueError("Stripe secret key not configured")

def create_customer(user_email: str, user_name: str, google_id: str) -> Dict:
    """Create a Stripe customer for the user."""
    initialize_stripe()
    
    try:
        customer = stripe.Customer.create(
            email=user_email,
            name=user_name,
            metadata={
                'google_id': google_id,
                'source': 'youtube_optimizer'
            }
        )
        
        logger.info(f"Created Stripe customer: {customer.id} for {user_email}")
        return {
            'customer_id': customer.id,
            'email': customer.email,
            'name': customer.name
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe customer creation failed: {e}")
        raise ValueError(f"Failed to create customer: {str(e)}")

def create_checkout_session(customer_id: str) -> Dict:
    """Create a Stripe Checkout Session for collecting payment method without charging."""
    initialize_stripe()
    
    try:
        from flask import current_app, url_for
        
        # Get the base URL for callbacks
        base_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        scheme = 'http' if current_app.config['DEBUG'] else 'https'
        
        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=['card'],
            mode='setup',  # Setup mode for saving payment methods without charging
            success_url=f"{scheme}://localhost:5001/stripe/checkout-success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{scheme}://localhost:5001/stripe/checkout-cancel",
            metadata={
                'purpose': 'youtube_optimizer_signup'
            },
            # Custom messaging to clarify no charging
            custom_text={
                'submit': {
                    'message': 'You will not be charged now. We\'re securely saving your payment method for future billing when you use our optimization services.'
                }
            },
            # Additional setup options
            payment_method_options={
                'card': {
                    'setup_future_usage': 'off_session'
                }
            }
        )
        
        logger.info(f"Created Checkout Session: {checkout_session.id} for customer: {customer_id}")
        return {
            'checkout_url': checkout_session.url,
            'session_id': checkout_session.id,
            'status': checkout_session.status
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Checkout Session creation failed: {e}")
        raise ValueError(f"Failed to create checkout session: {str(e)}")

def create_setup_intent(customer_id: str) -> Dict:
    """Create a Setup Intent for collecting payment method without charging."""
    initialize_stripe()
    
    try:
        setup_intent = stripe.SetupIntent.create(
            customer=customer_id,
            payment_method_types=['card'],
            usage='off_session',  # For future payments
            metadata={
                'purpose': 'youtube_optimizer_signup'
            }
        )
        
        logger.info(f"Created Setup Intent: {setup_intent.id} for customer: {customer_id}")
        return {
            'client_secret': setup_intent.client_secret,
            'setup_intent_id': setup_intent.id,
            'status': setup_intent.status
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Setup Intent creation failed: {e}")
        raise ValueError(f"Failed to create setup intent: {str(e)}")

def confirm_setup_intent(setup_intent_id: str) -> Dict:
    """Confirm and retrieve Setup Intent details."""
    initialize_stripe()
    
    try:
        setup_intent = stripe.SetupIntent.retrieve(setup_intent_id)
        
        result = {
            'id': setup_intent.id,
            'status': setup_intent.status,
            'customer_id': setup_intent.customer,
            'payment_method_id': setup_intent.payment_method,
            'succeeded': setup_intent.status == 'succeeded'
        }
        
        if result['succeeded']:
            # Get payment method details
            payment_method = stripe.PaymentMethod.retrieve(setup_intent.payment_method)
            result['card_details'] = {
                'brand': payment_method.card.brand,
                'last4': payment_method.card.last4,
                'exp_month': payment_method.card.exp_month,
                'exp_year': payment_method.card.exp_year
            }
            
            logger.info(f"Setup Intent confirmed: {setup_intent_id}, Payment Method: {setup_intent.payment_method}")
        
        return result
        
    except stripe.error.StripeError as e:
        logger.error(f"Setup Intent confirmation failed: {e}")
        raise ValueError(f"Failed to confirm setup intent: {str(e)}")

def get_customer_payment_methods(customer_id: str) -> Dict:
    """Get all payment methods for a customer."""
    initialize_stripe()
    
    try:
        payment_methods = stripe.PaymentMethod.list(
            customer=customer_id,
            type="card"
        )
        
        return {
            'payment_methods': [
                {
                    'id': pm.id,
                    'brand': pm.card.brand,
                    'last4': pm.card.last4,
                    'exp_month': pm.card.exp_month,
                    'exp_year': pm.card.exp_year
                }
                for pm in payment_methods.data
            ]
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Failed to retrieve payment methods: {e}")
        raise ValueError(f"Failed to get payment methods: {str(e)}")

def create_checkout_session_from_session() -> str:
    """Create Stripe Checkout Session using data from session after OAuth."""
    user_data = session.get('user_data')
    if not user_data:
        raise ValueError("No user data in session")
    
    user_email = user_data.get('email')
    user_name = user_data.get('name')
    google_id = user_data.get('sub')
    
    if not all([user_email, user_name, google_id]):
        raise ValueError("Incomplete user data in session")
    
    # Create or retrieve customer
    customer_info = create_customer(user_email, user_name, google_id)
    customer_id = customer_info['customer_id']
    
    # Create checkout session
    checkout_info = create_checkout_session(customer_id)
    
    # Store in session for later verification
    session['stripe_customer_id'] = customer_id
    session['stripe_checkout_session_id'] = checkout_info['session_id']
    session['payment_setup_initiated'] = True
    
    return checkout_info['checkout_url']

def process_checkout_session_success(session_id: str) -> Dict:
    """Process successful Stripe Checkout Session and get payment method details."""
    initialize_stripe()
    
    try:
        # Retrieve the checkout session
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        
        if checkout_session.status != 'complete':
            raise ValueError(f"Checkout session not complete: {checkout_session.status}")
        
        # Get the setup intent from the session
        setup_intent_id = checkout_session.setup_intent
        if not setup_intent_id:
            raise ValueError("No setup intent found in checkout session")
        
        # Get setup intent details
        setup_intent = stripe.SetupIntent.retrieve(setup_intent_id)
        
        result = {
            'checkout_session_id': session_id,
            'setup_intent_id': setup_intent_id,
            'customer_id': checkout_session.customer,
            'payment_method_id': setup_intent.payment_method,
            'succeeded': setup_intent.status == 'succeeded'
        }
        
        if result['succeeded']:
            # Get payment method details
            payment_method = stripe.PaymentMethod.retrieve(setup_intent.payment_method)
            result['card_details'] = {
                'brand': payment_method.card.brand,
                'last4': payment_method.card.last4,
                'exp_month': payment_method.card.exp_month,
                'exp_year': payment_method.card.exp_year
            }
            
            logger.info(f"Checkout session successful: {session_id}, Payment Method: {setup_intent.payment_method}")
        
        return result
        
    except stripe.error.StripeError as e:
        logger.error(f"Checkout session processing failed: {e}")
        raise ValueError(f"Failed to process checkout session: {str(e)}")

def process_payment_setup_from_session() -> Tuple[str, str]:
    """Process payment setup using data from session after OAuth."""
    user_data = session.get('user_data')
    if not user_data:
        raise ValueError("No user data in session")
    
    user_email = user_data.get('email')
    user_name = user_data.get('name')
    google_id = user_data.get('sub')
    
    if not all([user_email, user_name, google_id]):
        raise ValueError("Incomplete user data in session")
    
    # Create or retrieve customer
    customer_info = create_customer(user_email, user_name, google_id)
    customer_id = customer_info['customer_id']
    
    # Create setup intent
    setup_info = create_setup_intent(customer_id)
    
    # Store in session for later verification
    session['stripe_customer_id'] = customer_id
    session['stripe_setup_intent_id'] = setup_info['setup_intent_id']
    session['payment_setup_initiated'] = True
    
    return setup_info['client_secret'], customer_id