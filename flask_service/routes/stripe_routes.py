import logging
from flask import Blueprint, request, jsonify, redirect, current_app, session
from services.stripe_service import (
    create_checkout_session_from_session,
    process_checkout_session_success,
    get_customer_payment_methods,
    find_customer_by_email,
)
from services.auth_service import AuthService

logger = logging.getLogger(__name__)
stripe_bp = Blueprint('stripe', __name__)

# Simple HTML template for Stripe Elements
STRIPE_CHECKOUT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Add Payment Method - YouTube Optimizer</title>
    <script src="https://js.stripe.com/v3/"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; }
        .form-row { margin: 20px 0; }
        #card-element { padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
        #card-errors { color: #fa755a; margin-top: 10px; }
        button { background: #5469d4; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .loading { display: none; }
    </style>
</head>
<body>
    <h2>Add Payment Method</h2>
    <p>Please add a valid credit card to continue with your YouTube optimization service.</p>
    
    <form id="setup-form">
        <div class="form-row">
            <div id="card-element">
                <!-- Stripe Elements will create form elements here -->
            </div>
            <div id="card-errors" role="alert"></div>
        </div>
        <button id="submit-button">
            <span id="button-text">Add Payment Method</span>
            <span id="spinner" class="loading">...</span>
        </button>
    </form>

    <script>
        var stripe = Stripe('{{ stripe_public_key }}');
        var elements = stripe.elements();
        var cardElement = elements.create('card');
        cardElement.mount('#card-element');

        var form = document.getElementById('setup-form');
        var submitButton = document.getElementById('submit-button');
        var buttonText = document.getElementById('button-text');
        var spinner = document.getElementById('spinner');

        form.addEventListener('submit', function(event) {
            event.preventDefault();
            
            submitButton.disabled = true;
            buttonText.style.display = 'none';
            spinner.style.display = 'inline';

            stripe.confirmCardSetup('{{ client_secret }}', {
                payment_method: {
                    card: cardElement,
                }
            }).then(function(result) {
                if (result.error) {
                    document.getElementById('card-errors').textContent = result.error.message;
                    submitButton.disabled = false;
                    buttonText.style.display = 'inline';
                    spinner.style.display = 'none';
                } else {
                    // Payment method successfully added
                    window.location.href = '/stripe/payment-success?setup_intent=' + result.setupIntent.id;
                }
            });
        });

        cardElement.on('change', function(event) {
            var displayError = document.getElementById('card-errors');
            if (event.error) {
                displayError.textContent = event.error.message;
            } else {
                displayError.textContent = '';
            }
        });
    </script>
</body>
</html>
"""

@stripe_bp.route('/payment-setup')
def payment_setup():
    """Redirect to Stripe Checkout for payment method collection."""
    try:
        # Check if user completed OAuth
        user_data = session.get('user_data')
        if not user_data:
            logger.error("No user data in session for payment setup")
            frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
            return redirect(f"{frontend_url}?auth=failed&reason=session_expired")
        
        # Check if payment already set up
        if session.get('payment_method_validated'):
            logger.info("Payment method already validated, redirecting to success")
            return redirect('/stripe/checkout-success')

        # Check if user email exists in Stripe customers
        user_email = user_data.get('email')
        existing_customer = find_customer_by_email(user_email)
        if existing_customer:
            logger.info(f"Found existing customer for {user_email}: {existing_customer['customer_id']}")
            
            # Check if customer has payment methods
            try:
                payment_methods = get_customer_payment_methods(existing_customer['customer_id'])
                if payment_methods['payment_methods']:
                    logger.info(f"Customer {existing_customer['customer_id']} already has payment methods")
                    # Customer exists and has payment methods, redirect to frontend
                    session['stripe_customer_id'] = existing_customer['customer_id']
                    session['payment_method_validated'] = True
                    frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
                    print(frontend_url)
                    return redirect(f"{frontend_url}?auth=success&existing_customer=true")
                else:
                    logger.info(f"Customer {existing_customer['customer_id']} exists but has no payment methods")
                    # Customer exists but no payment methods, proceed to add payment method
                    session['stripe_customer_id'] = existing_customer['customer_id']
            except Exception as e:
                logger.error(f"Error checking payment methods for existing customer: {e}")
                # If we can't check payment methods, proceed to add payment method
                session['stripe_customer_id'] = existing_customer['customer_id']
        else:
            logger.info(f"No existing customer found for {user_email}, will create new customer")

        # Create Stripe Checkout Session to add payment method
        checkout_url = create_checkout_session_from_session()
        
        logger.info(f"Stripe Checkout initiated for user: {user_data.get('email')}")
        logger.info(f"Redirecting to: {checkout_url}")
        
        # Redirect to Stripe Checkout
        return redirect(checkout_url)
        
    except ValueError as e:
        logger.error(f"Payment setup validation error: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=payment_setup_error")
    except Exception as e:
        logger.error(f"Unexpected error in payment setup: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=unexpected_error")

@stripe_bp.route('/payment-success')
def payment_success():
    """Handle successful payment method setup."""
    try:
        setup_intent_id = request.args.get('setup_intent')
        if not setup_intent_id:
            raise ValueError("No setup intent ID provided")
        
        # Verify the setup intent matches session
        session_setup_intent_id = session.get('stripe_setup_intent_id')
        if setup_intent_id != session_setup_intent_id:
            raise ValueError("Setup intent ID mismatch")
        
        # Confirm the setup intent
        setup_result = confirm_setup_intent(setup_intent_id)
        
        if not setup_result['succeeded']:
            raise ValueError(f"Setup intent not successful: {setup_result['status']}")
        
        # Store payment method info in session
        session['payment_method_validated'] = True
        session['stripe_payment_method_id'] = setup_result['payment_method_id']
        session['card_details'] = setup_result['card_details']
        
        logger.info(f"Payment method validated successfully: {setup_result['payment_method_id']}")
        
        # Now trigger the actual user authentication and video processing
        try:
            # Create API payload from session (same as OAuth flow)
            from services.oauth_service import OAuthService
            api_payload = OAuthService.create_api_payload()
            
            # Add Stripe information to payload
            api_payload['stripe'] = {
                'customer_id': session.get('stripe_customer_id'),
                'payment_method_id': setup_result['payment_method_id'],
                'card_details': setup_result['card_details']
            }
            
            # Process through existing auth service (this triggers video processing)
            result = AuthService.process_user_authentication(api_payload)
            
            logger.info(f"User authentication with payment processed successfully: {result}")
            
            # Redirect to frontend with success
            frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
            return redirect(f"{frontend_url}?auth=success")
            
        except Exception as e:
            logger.error(f"Error processing user authentication after payment: {e}")
            frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
            return redirect(f"{frontend_url}?auth=failed&reason=processing_error")
        
    except ValueError as e:
        logger.error(f"Payment success validation error: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=payment_validation_error")
    except Exception as e:
        logger.error(f"Unexpected error in payment success: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=unexpected_error")

@stripe_bp.route('/checkout-success')
def checkout_success():
    """Handle successful Stripe Checkout Session."""
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            raise ValueError("No session ID provided")
        
        # Verify the session matches what we stored
        stored_session_id = session.get('stripe_checkout_session_id')
        if session_id != stored_session_id:
            raise ValueError("Session ID mismatch")
        
        # Process the checkout session
        checkout_result = process_checkout_session_success(session_id)
        
        if not checkout_result['succeeded']:
            raise ValueError(f"Checkout session not successful")
        
        # Store payment method info in session
        session['payment_method_validated'] = True
        session['stripe_payment_method_id'] = checkout_result['payment_method_id']
        session['card_details'] = checkout_result['card_details']
        
        logger.info(f"Payment method validated via Checkout: {checkout_result['payment_method_id']}")
        
        # Now trigger the actual user authentication and video processing
        try:
            # Create API payload from session (same as OAuth flow)
            from services.oauth_service import OAuthService
            api_payload = OAuthService.create_api_payload()
            
            # Add Stripe information to payload
            api_payload['stripe'] = {
                'customer_id': checkout_result['customer_id'],
                'payment_method_id': checkout_result['payment_method_id'],
                'card_details': checkout_result['card_details']
            }
            
            # Process through existing auth service (this triggers video processing)
            result = AuthService.process_user_authentication(api_payload)
            
            logger.info(f"User authentication with payment processed successfully: {result}")
            
            # Redirect to frontend with success
            frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
            return redirect(f"{frontend_url}?auth=success")
            
        except Exception as e:
            logger.error(f"Error processing user authentication after payment: {e}")
            frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
            return redirect(f"{frontend_url}?auth=failed&reason=processing_error")
        
    except ValueError as e:
        logger.error(f"Checkout success validation error: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=checkout_validation_error")
    except Exception as e:
        logger.error(f"Unexpected error in checkout success: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=unexpected_error")

@stripe_bp.route('/checkout-cancel')
def checkout_cancel():
    """Handle cancelled Stripe Checkout Session."""
    logger.info("User cancelled Stripe Checkout")
    frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
    return redirect(f"{frontend_url}?auth=cancelled&reason=payment_cancelled")

@stripe_bp.route('/payment-cancel')
def payment_cancel():
    """Handle payment method setup cancellation."""
    logger.info("User cancelled payment setup")
    frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
    return redirect(f"{frontend_url}?auth=cancelled&reason=payment_cancelled")

@stripe_bp.route('/payment-status')
def payment_status():
    """Get current payment method status."""
    try:
        customer_id = session.get('stripe_customer_id')
        payment_validated = session.get('payment_method_validated', False)
        
        if not customer_id:
            return jsonify({
                'payment_method_validated': False,
                'has_customer': False
            })
        
        # Get payment methods if customer exists
        payment_methods = get_customer_payment_methods(customer_id)
        
        return jsonify({
            'payment_method_validated': payment_validated,
            'has_customer': True,
            'customer_id': customer_id,
            'payment_methods': payment_methods['payment_methods'],
            'card_details': session.get('card_details')
        })
        
    except Exception as e:
        logger.error(f"Error getting payment status: {e}")
        return jsonify({'error': 'Failed to get payment status'}), 500

@stripe_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events."""
    import stripe
    from services.stripe_service import initialize_stripe
    
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        initialize_stripe()
        # Verify webhook signature if STRIPE_WEBHOOK_SECRET is configured
        webhook_secret = current_app.config.get('STRIPE_WEBHOOK_SECRET')
        if webhook_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        else:
            # If no webhook secret configured, just parse the event
            import json
            event = json.loads(payload)
        
    except ValueError as e:
        logger.error(f"Invalid webhook payload: {e}")
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid webhook signature: {e}")
        return 'Invalid signature', 400
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return 'Webhook error', 400
    
    # Handle the event
    event_type = event.get('type')
    logger.info(f"Received Stripe webhook: {event_type}")
    
    if event_type == 'setup_intent.succeeded':
        setup_intent = event['data']['object']
        logger.info(f"Setup intent succeeded: {setup_intent['id']}")
        
    elif event_type == 'setup_intent.setup_failed':
        setup_intent = event['data']['object']
        logger.warning(f"Setup intent failed: {setup_intent['id']}")
        
    elif event_type == 'payment_method.attached':
        payment_method = event['data']['object']
        logger.info(f"Payment method attached: {payment_method['id']}")
        
    elif event_type == 'customer.created':
        customer = event['data']['object']
        logger.info(f"Customer created: {customer['id']}")
        
    else:
        logger.info(f"Unhandled webhook event type: {event_type}")
    
    return 'Success', 200