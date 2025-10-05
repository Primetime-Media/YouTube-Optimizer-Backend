# services/stripe_service.py
"""
Production-Ready Stripe Integration Service
Handles payment processing, subscriptions, and billing
"""

import stripe
from typing import Dict, Optional, List, Any
from decimal import Decimal
from datetime import datetime, timezone
import logging
import json

from utils.db import get_db_connection, execute_query, transaction
from config import settings

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.stripe_secret_key.get_secret_value()
stripe.api_version = "2023-10-16"  # Pin API version for stability

# Webhook secret
WEBHOOK_SECRET = settings.stripe_webhook_secret.get_secret_value()

# Platform fee percentage (30%)
PLATFORM_FEE_PERCENTAGE = Decimal('0.30')


# ============================================================================
# STRIPE CONFIGURATION
# ============================================================================

class StripeConfig:
    """Stripe API configuration and constants"""
    
    # Products (create these in Stripe Dashboard)
    PRODUCT_STARTER = "prod_starter"
    PRODUCT_GROWTH = "prod_growth"
    PRODUCT_SCALE = "prod_scale"
    PRODUCT_ENTERPRISE = "prod_enterprise"
    
    # Prices (create these in Stripe Dashboard)
    PRICE_STARTER = "price_starter_monthly"
    PRICE_GROWTH = "price_growth_monthly"
    PRICE_SCALE = "price_scale_monthly"
    PRICE_ENTERPRISE = "price_enterprise_monthly"
    
    @staticmethod
    def get_price_id(plan: str) -> Optional[str]:
        """Get Stripe price ID for a plan"""
        price_map = {
            'starter': StripeConfig.PRICE_STARTER,
            'growth': StripeConfig.PRICE_GROWTH,
            'scale': StripeConfig.PRICE_SCALE,
            'enterprise': StripeConfig.PRICE_ENTERPRISE,
        }
        return price_map.get(plan.lower())
    
    @staticmethod
    def get_webhook_secret() -> str:
        """Get webhook secret"""
        return WEBHOOK_SECRET


# ============================================================================
# CUSTOMER MANAGEMENT
# ============================================================================

def find_customer_by_email(email: str) -> Optional[stripe.Customer]:
    """
    Find existing Stripe customer by email
    
    Args:
        email: Customer email address
        
    Returns:
        Stripe Customer object or None if not found
    """
    try:
        customers = stripe.Customer.list(email=email, limit=1)
        return customers.data[0] if customers.data else None
    except stripe.error.StripeError as e:
        logger.error(f"Error finding customer by email {email}: {e}")
        return None


def create_customer(
    email: str,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None
) -> Optional[stripe.Customer]:
    """
    Create a new Stripe customer
    
    Args:
        email: Customer email
        name: Customer name
        metadata: Additional metadata
        
    Returns:
        Stripe Customer object or None on error
    """
    try:
        customer = stripe.Customer.create(
            email=email,
            name=name,
            metadata=metadata or {}
        )
        
        logger.info(f"Created Stripe customer: {customer.id} for {email}")
        return customer
        
    except stripe.error.StripeError as e:
        logger.error(f"Error creating Stripe customer for {email}: {e}")
        return None


def get_or_create_customer(
    email: str,
    name: Optional[str] = None,
    user_id: Optional[int] = None
) -> Optional[stripe.Customer]:
    """
    Get existing customer or create new one
    
    Args:
        email: Customer email
        name: Customer name
        user_id: Internal user ID for metadata
        
    Returns:
        Stripe Customer object or None on error
    """
    # Check for existing customer
    customer = find_customer_by_email(email)
    
    if customer:
        logger.info(f"Found existing Stripe customer: {customer.id}")
        return customer
    
    # Create new customer
    metadata = {}
    if user_id:
        metadata['user_id'] = str(user_id)
    
    return create_customer(email, name, metadata)


# ============================================================================
# PAYMENT METHOD MANAGEMENT
# ============================================================================

def create_setup_intent(customer_id: str) -> Optional[stripe.SetupIntent]:
    """
    Create a SetupIntent for collecting payment method
    
    Args:
        customer_id: Stripe customer ID
        
    Returns:
        SetupIntent object or None on error
    """
    try:
        setup_intent = stripe.SetupIntent.create(
            customer=customer_id,
            payment_method_types=['card'],
            usage='off_session'  # For future charges
        )
        
        logger.info(f"Created SetupIntent for customer {customer_id}")
        return setup_intent
        
    except stripe.error.StripeError as e:
        logger.error(f"Error creating SetupIntent for {customer_id}: {e}")
        return None


def confirm_setup_intent(
    setup_intent_id: str,
    payment_method_id: str
) -> Optional[stripe.SetupIntent]:
    """
    Confirm a SetupIntent with payment method
    
    Args:
        setup_intent_id: SetupIntent ID
        payment_method_id: Payment method ID
        
    Returns:
        Confirmed SetupIntent or None on error
    """
    try:
        setup_intent = stripe.SetupIntent.confirm(
            setup_intent_id,
            payment_method=payment_method_id
        )
        
        logger.info(f"Confirmed SetupIntent {setup_intent_id}")
        return setup_intent
        
    except stripe.error.StripeError as e:
        logger.error(f"Error confirming SetupIntent {setup_intent_id}: {e}")
        return None


def get_customer_payment_methods(
    customer_id: str
) -> List[stripe.PaymentMethod]:
    """
    Get all payment methods for a customer
    
    Args:
        customer_id: Stripe customer ID
        
    Returns:
        List of PaymentMethod objects
    """
    try:
        payment_methods = stripe.PaymentMethod.list(
            customer=customer_id,
            type='card'
        )
        return payment_methods.data
    except stripe.error.StripeError as e:
        logger.error(f"Error getting payment methods for {customer_id}: {e}")
        return []


# ============================================================================
# SUBSCRIPTION MANAGEMENT
# ============================================================================

def create_subscription(
    customer_id: str,
    price_id: str,
    trial_days: int = 0,
    metadata: Optional[Dict[str, str]] = None
) -> Optional[stripe.Subscription]:
    """
    Create a subscription for a customer
    
    Args:
        customer_id: Stripe customer ID
        price_id: Stripe price ID
        trial_days: Number of trial days
        metadata: Additional metadata
        
    Returns:
        Subscription object or None on error
    """
    try:
        subscription_params = {
            'customer': customer_id,
            'items': [{'price': price_id}],
            'metadata': metadata or {},
            'payment_behavior': 'default_incomplete',
            'payment_settings': {
                'save_default_payment_method': 'on_subscription'
            },
            'expand': ['latest_invoice.payment_intent']
        }
        
        if trial_days > 0:
            subscription_params['trial_period_days'] = trial_days
        
        subscription = stripe.Subscription.create(**subscription_params)
        
        logger.info(
            f"Created subscription {subscription.id} for customer {customer_id}"
        )
        return subscription
        
    except stripe.error.StripeError as e:
        logger.error(f"Error creating subscription for {customer_id}: {e}")
        return None


def cancel_subscription(
    subscription_id: str,
    at_period_end: bool = True
) -> Optional[stripe.Subscription]:
    """
    Cancel a subscription
    
    Args:
        subscription_id: Subscription ID
        at_period_end: Cancel at period end vs immediately
        
    Returns:
        Cancelled subscription or None on error
    """
    try:
        if at_period_end:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
        else:
            subscription = stripe.Subscription.delete(subscription_id)
        
        logger.info(f"Cancelled subscription {subscription_id}")
        return subscription
        
    except stripe.error.StripeError as e:
        logger.error(f"Error cancelling subscription {subscription_id}: {e}")
        return None


def update_subscription(
    subscription_id: str,
    new_price_id: str
) -> Optional[stripe.Subscription]:
    """
    Update subscription to new plan
    
    Args:
        subscription_id: Subscription ID
        new_price_id: New price ID
        
    Returns:
        Updated subscription or None on error
    """
    try:
        subscription = stripe.Subscription.retrieve(subscription_id)
        
        subscription = stripe.Subscription.modify(
            subscription_id,
            items=[{
                'id': subscription['items']['data'][0].id,
                'price': new_price_id,
            }],
            proration_behavior='create_prorations'
        )
        
        logger.info(f"Updated subscription {subscription_id} to price {new_price_id}")
        return subscription
        
    except stripe.error.StripeError as e:
        logger.error(f"Error updating subscription {subscription_id}: {e}")
        return None


# ============================================================================
# CHECKOUT SESSION
# ============================================================================

def create_checkout_session(
    customer_id: str,
    price_id: str,
    success_url: str,
    cancel_url: str,
    trial_days: int = 0,
    metadata: Optional[Dict[str, str]] = None
) -> Optional[stripe.checkout.Session]:
    """
    Create a Checkout Session for subscription
    
    Args:
        customer_id: Stripe customer ID
        price_id: Price ID
        success_url: Redirect URL on success
        cancel_url: Redirect URL on cancel
        trial_days: Trial period days
        metadata: Additional metadata
        
    Returns:
        Checkout Session or None on error
    """
    try:
        session_params = {
            'customer': customer_id,
            'mode': 'subscription',
            'line_items': [{
                'price': price_id,
                'quantity': 1
            }],
            'success_url': success_url,
            'cancel_url': cancel_url,
            'metadata': metadata or {}
        }
        
        if trial_days > 0:
            session_params['subscription_data'] = {
                'trial_period_days': trial_days
            }
        
        session = stripe.checkout.Session.create(**session_params)
        
        logger.info(f"Created checkout session {session.id} for customer {customer_id}")
        return session
        
    except stripe.error.StripeError as e:
        logger.error(f"Error creating checkout session for {customer_id}: {e}")
        return None


# ============================================================================
# INVOICE & PAYMENT
# ============================================================================

def create_invoice_item(
    customer_id: str,
    amount: Decimal,
    description: str,
    metadata: Optional[Dict[str, str]] = None
) -> Optional[stripe.InvoiceItem]:
    """
    Create an invoice item for one-time charges
    
    Args:
        customer_id: Customer ID
        amount: Amount in dollars
        description: Item description
        metadata: Additional metadata
        
    Returns:
        InvoiceItem or None on error
    """
    try:
        # Convert to cents
        amount_cents = int(amount * 100)
        
        invoice_item = stripe.InvoiceItem.create(
            customer=customer_id,
            amount=amount_cents,
            currency='usd',
            description=description,
            metadata=metadata or {}
        )
        
        logger.info(f"Created invoice item for customer {customer_id}: ${amount}")
        return invoice_item
        
    except stripe.error.StripeError as e:
        logger.error(f"Error creating invoice item for {customer_id}: {e}")
        return None


def create_and_pay_invoice(
    customer_id: str,
    auto_advance: bool = True
) -> Optional[stripe.Invoice]:
    """
    Create and finalize an invoice
    
    Args:
        customer_id: Customer ID
        auto_advance: Auto-finalize and pay
        
    Returns:
        Invoice object or None on error
    """
    try:
        invoice = stripe.Invoice.create(
            customer=customer_id,
            auto_advance=auto_advance,
            collection_method='charge_automatically'
        )
        
        if auto_advance:
            invoice = stripe.Invoice.finalize_invoice(invoice.id)
            invoice = stripe.Invoice.pay(invoice.id)
        
        logger.info(f"Created and finalized invoice {invoice.id} for {customer_id}")
        return invoice
        
    except stripe.error.StripeError as e:
        logger.error(f"Error creating invoice for {customer_id}: {e}")
        return None


# ============================================================================
# WEBHOOK HANDLING
# ============================================================================

def verify_webhook_signature(
    payload: bytes,
    sig_header: str
) -> Optional[Dict]:
    """
    Verify Stripe webhook signature and construct event
    
    Args:
        payload: Raw request body
        sig_header: Stripe-Signature header value
        
    Returns:
        Verified event dict or None on error
    """
    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            WEBHOOK_SECRET
        )
        
        logger.info(f"Verified webhook event: {event['type']}")
        return event
        
    except ValueError as e:
        logger.error(f"Invalid webhook payload: {e}")
        return None
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid webhook signature: {e}")
        return None


def handle_webhook_event(event: Dict) -> bool:
    """
    Handle Stripe webhook events
    
    Args:
        event: Verified Stripe event
        
    Returns:
        True if handled successfully, False otherwise
    """
    event_type = event['type']
    event_data = event['data']['object']
    
    logger.info(f"Handling webhook event: {event_type}")
    
    try:
        if event_type == 'customer.subscription.created':
            return _handle_subscription_created(event_data)
        
        elif event_type == 'customer.subscription.updated':
            return _handle_subscription_updated(event_data)
        
        elif event_type == 'customer.subscription.deleted':
            return _handle_subscription_deleted(event_data)
        
        elif event_type == 'invoice.payment_succeeded':
            return _handle_payment_succeeded(event_data)
        
        elif event_type == 'invoice.payment_failed':
            return _handle_payment_failed(event_data)
        
        elif event_type == 'customer.updated':
            return _handle_customer_updated(event_data)
        
        else:
            logger.info(f"Unhandled webhook event type: {event_type}")
            return True  # Not an error, just not handled
        
    except Exception as e:
        logger.error(f"Error handling webhook {event_type}: {e}", exc_info=True)
        return False


def _handle_subscription_created(subscription: Dict) -> bool:
    """Handle subscription.created webhook"""
    try:
        customer_id = subscription['customer']
        subscription_id = subscription['id']
        status = subscription['status']
        
        # Update database
        with transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET stripe_subscription_id = %s,
                        subscription_status = %s,
                        updated_at = NOW()
                    WHERE stripe_customer_id = %s
                """, (subscription_id, status, customer_id))
        
        logger.info(f"Subscription created: {subscription_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error handling subscription created: {e}", exc_info=True)
        return False


def _handle_subscription_updated(subscription: Dict) -> bool:
    """Handle subscription.updated webhook"""
    try:
        subscription_id = subscription['id']
        status = subscription['status']
        
        with transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET subscription_status = %s,
                        updated_at = NOW()
                    WHERE stripe_subscription_id = %s
                """, (status, subscription_id))
        
        logger.info(f"Subscription updated: {subscription_id} -> {status}")
        return True
        
    except Exception as e:
        logger.error(f"Error handling subscription updated: {e}", exc_info=True)
        return False


def _handle_subscription_deleted(subscription: Dict) -> bool:
    """Handle subscription.deleted webhook"""
    try:
        subscription_id = subscription['id']
        
        with transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET subscription_status = 'canceled',
                        updated_at = NOW()
                    WHERE stripe_subscription_id = %s
                """, (subscription_id,))
        
        logger.info(f"Subscription deleted: {subscription_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error handling subscription deleted: {e}", exc_info=True)
        return False


def _handle_payment_succeeded(invoice: Dict) -> bool:
    """Handle invoice.payment_succeeded webhook"""
    try:
        customer_id = invoice['customer']
        amount_paid = Decimal(invoice['amount_paid']) / 100  # Convert from cents
        
        logger.info(f"Payment succeeded for {customer_id}: ${amount_paid}")
        
        # Record payment in database
        with transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO payments (
                        stripe_customer_id,
                        stripe_invoice_id,
                        amount,
                        status,
                        created_at
                    ) VALUES (%s, %s, %s, %s, NOW())
                """, (customer_id, invoice['id'], amount_paid, 'succeeded'))
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling payment succeeded: {e}", exc_info=True)
        return False


def _handle_payment_failed(invoice: Dict) -> bool:
    """Handle invoice.payment_failed webhook"""
    try:
        customer_id = invoice['customer']
        
        logger.warning(f"Payment failed for customer {customer_id}")
        
        # Update user status or send notification
        # Implementation depends on business logic
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling payment failed: {e}", exc_info=True)
        return False


def _handle_customer_updated(customer: Dict) -> bool:
    """Handle customer.updated webhook"""
    try:
        customer_id = customer['id']
        email = customer['email']
        
        with transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET email = %s,
                        updated_at = NOW()
                    WHERE stripe_customer_id = %s
                """, (email, customer_id))
        
        logger.info(f"Customer updated: {customer_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error handling customer updated: {e}", exc_info=True)
        return False


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'StripeConfig',
    'find_customer_by_email',
    'create_customer',
    'get_or_create_customer',
    'create_setup_intent',
    'confirm_setup_intent',
    'get_customer_payment_methods',
    'create_subscription',
    'cancel_subscription',
    'update_subscription',
    'create_checkout_session',
    'create_invoice_item',
    'create_and_pay_invoice',
    'verify_webhook_signature',
    'handle_webhook_event',
]
