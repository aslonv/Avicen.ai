import logging
import json
from flask import Blueprint, request, jsonify, current_app
from .decorators.security import signature_required
from .utils.whatsapp_utils import is_valid_whatsapp_message
from .services.medical_service import query_lm, read_bloodtest

webhook_blueprint = Blueprint("webhook", __name__)

def process_whatsapp_message(body):
   """Process WhatsApp messages and generate responses using medical model"""
   message = body['entry'][0]['changes'][0]['value']['messages'][0]
   message_body = message['text']['body']
   wa_id = message['from']
   name = message.get('from', {}).get('name', 'User')
   
   if "blood test" in message_body.lower():
       response = read_bloodtest(message_body)
   else:
       response = query_lm(message_body, wa_id)
   
   return response

def handle_message():
   """
   Handle incoming webhook events from the WhatsApp API.
   Every message send will trigger 4 HTTP requests: message, sent, delivered, read.
   Returns: tuple containing JSON response and HTTP status code.
   """
   body = request.get_json()

   if (body.get("entry", [{}])[0]
       .get("changes", [{}])[0]
       .get("value", {})
       .get("statuses")):
       logging.info("Received a WhatsApp status update.")
       return jsonify({"status": "ok"}), 200

   try:
       if is_valid_whatsapp_message(body):
           response = process_whatsapp_message(body)
           return jsonify({"status": "ok", "response": response}), 200
       else:
           return (
               jsonify({"status": "error", "message": "Not a WhatsApp API event"}),
               404,
           )
   except json.JSONDecodeError:
       logging.error("Failed to decode JSON")
       return jsonify({"status": "error", "message": "Invalid JSON provided"}), 400

def verify():
   """Required webhook verification for WhatsApp"""
   mode = request.args.get("hub.mode")
   token = request.args.get("hub.verify_token")
   challenge = request.args.get("hub.challenge")
   
   if mode and token:
       if mode == "subscribe" and token == current_app.config["VERIFY_TOKEN"]:
           logging.info("WEBHOOK_VERIFIED")
           return challenge, 200
       else:
           logging.info("VERIFICATION_FAILED")
           return jsonify({"status": "error", "message": "Verification failed"}), 403
   else:
       logging.info("MISSING_PARAMETER")
       return jsonify({"status": "error", "message": "Missing parameters"}), 400

@webhook_blueprint.route("/webhook", methods=["GET"])
def webhook_get():
   return verify()

@webhook_blueprint.route("/webhook", methods=["POST"])
@signature_required
def webhook_post():
   return handle_message()

@webhook_blueprint.route("/", methods=["GET"])
def home():
   return "WhatsApp Webhook is running", 200