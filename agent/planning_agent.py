import os
from typing import Dict, Type, Annotated, TypedDict
import logging
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from agent.router_agent import planning_route_query
from agent.generic_agent import process_generic_query
from agent.product_review_agent import setup_product_review_agent
from agent.composer_agent import compose_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an efficient and helpful AI planning agent. Do not assume anything. Always use route_query to determine the router_response. The router_response is the variable which has the user query type. Always refer to this variable router_response.
If router_response is 'generic', use handle_generic_query to process the user query and then use compose_response to format the response. 
If router_response is 'product_review', use get_product_info to retrieve product-related data and then use compose_response to format the response.

Example:

    User: Okay , i want to buy a phone. What buying options do you have ?
    Thought: I will use route_query to understand if query is product_review or generic
    Action: route_query
    Observation: product_review
    Thought: This query is actually a product review request, not a generic query. I will use get_product_info to get appropriate response.
    Action: get_product_info
    Action Input: User query: Okay , i want to buy a phone . What options do you have ?
    Observation: ok
    Thought:I have got the final answer. I will use compose_responses to format the response.
    Action: compose_responses
    Final Answer: ok
    """


def get_product_info(state: Dict, config: dict) -> Dict:
    """Handle product-related queries using ProductReviewAgent"""
    try:
        product_agent = setup_product_review_agent()
        response = product_agent.process_review_query(state, config)
        
        if "error" in response:
            return {"error": response["error"]}
            
        return {"product_info": response["review_response"]}
        
    except Exception as e:
        logger.error(f"Error in get_product_info: {e}")
        return {"error": str(e)}


def prepare_response_for_composer(state: Dict, config: dict) -> Dict:
    """Prepare the response data for the composer agent"""
    thread_id = config["configurable"]["thread_id"]
    logger.info(f"Preparing response for thread {thread_id}")
    
    try:
        # Extract the response text based on query type
        response_text = ""
        if "product_info" in state:
            response_text = state["product_info"]
        elif "generic_response" in state:
            response_text = state["generic_response"]
            print("*****response_text****", response_text)
        else:
            response_text = "I apologize, but I couldn't process your request properly. Please try again."
        
        # Create a temporary state for composer
        composer_state = {
            "messages": state["messages"],
            "response_text": response_text
        }

        # Call compose_response with state and config
        composed_state = compose_response(composer_state, config)
        
        # Selectively update state
        state["final_response"] = composed_state.get("final_response", "")
        # if "messages" in composed_state:
        #     state["messages"].extend(composed_state["messages"])
        
        return state
        
    except Exception as e:
        logger.error(f"Error in prepare_response_for_composer: {e}")
        error_message = "I apologize, but I encountered an error. Please try again."
        state["final_response"] = error_message
        state["messages"].append(AIMessage(content=error_message))
        return state


def route_next_step(state: Dict) -> str:
    """Route to next step based on router_response"""
    router_response = state.get("router_response", "")
    if router_response == "product_review":
        return "product"
    return "generic"


# In the below function: StateGraph takes the State class as a parameter to:
#       Know how to instantiate new state objects
#       Understand the structure of the state
#       Validate state transitions
#       Manage state typing throughout the workflow


def setup_agent_graph(State: Type) -> tuple[StateGraph, MemorySaver]:
    """Setup and return the agent workflow graph"""
    memory = MemorySaver()
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("route_query", planning_route_query)
    workflow.add_node("get_product_info", get_product_info)
    workflow.add_node("handle_generic_query", process_generic_query)
    workflow.add_node("prepare_response", prepare_response_for_composer)
    
    # Add conditional edges from route_query
    workflow.add_conditional_edges(
        "route_query",
        route_next_step,
        {
            "product": "get_product_info",
            "generic": "handle_generic_query"
        }
    )
    
    # Add regular edges
    workflow.add_edge("get_product_info", "prepare_response")
    workflow.add_edge("handle_generic_query", "prepare_response")
    workflow.add_edge("prepare_response", END)
    
    # Set entry point
    workflow.add_edge(START, "route_query")
    
    compiled_workflow = workflow.compile(checkpointer=memory)

    return compiled_workflow, memory