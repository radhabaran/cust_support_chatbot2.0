# router_agent.py
from typing import Dict
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

anthro_api_key = os.environ['ANTHRO_KEY']           
os.environ['ANTHROPIC_API_KEY'] = anthro_api_key

# Initialize LLM
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm = ChatAnthropic(model="claude-3-haiku-20240307")


class RouterResponse:
    PRODUCT_REVIEW = "product_review"
    GENERIC = "generic"


# def should_process_product_review(state: Dict) -> bool:
#     """Determine if query should be routed to product review handler"""
#     return state.get("router_response") == RouterResponse.PRODUCT_REVIEW


# def should_process_generic(state: Dict) -> bool:
#     """Determine if query should be routed to generic handler"""
#     return state.get("router_response") == RouterResponse.GENERIC


def extract_current_message(state: Dict) -> str:
    """Extract the current message from state"""
    if "current_message" in state:
        return state["current_message"]
    
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            return last_message.content
    return ""


def planning_route_query(state: Dict, config: Dict) -> Dict:
    """Route the query based on content analysis"""
    try:
        # print("Debug - Config:", config)
        # Extract current message from state
        current_message = extract_current_message(state)

        prompt = f"""Analyze the following query and determine if it's related to product review or a generic query.
        
        Product Review queries include:
        - Questions about product features, specifications, or capabilities
        - Product prices and availability inquiries
        - Requests for product reviews or comparisons
        - Product warranty or guarantee questions
        - Product shipping or delivery inquiries
        - Product compatibility or dimension questions
        - Product recommendations
        
        Generic queries include:
        - Customer service inquiries
        - Account-related questions
        - Technical support issues
        - Website navigation help
        - Payment or billing queries
        - Return policy questions
        - Company information requests
        
        Query: {current_message}
        
        Return ONLY 'product_review' or 'generic' as response."""
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages, config).content.lower().strip()
        
        category = RouterResponse.PRODUCT_REVIEW if RouterResponse.PRODUCT_REVIEW in response else RouterResponse.GENERIC
        
        # Log the routing decision
        logger.info(f"Routed message: '{current_message[:50]}...' to category: {category}")

        return {
            "router_response": category,
            "routing_metadata": {
                "routing_category": category,
                "original_message": current_message[:100]  # First 100 chars for context
            }
        }
    
    except Exception as e:
        logger.error(f"Error in planning_route_query: {e}")
        return {
            "router_response": RouterResponse.GENERIC,
            "routing_metadata": {
                "routing_category": RouterResponse.GENERIC,
                "error": str(e)
            }
        }