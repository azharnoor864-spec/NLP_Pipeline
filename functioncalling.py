import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

# =====================================================================
# STEP 1: DEFINE LOCAL PYTHON FUNCTIONS (The Active Workers)
# =====================================================================

def get_current_time(timezone="UTC"):
    """Returns the current system date and time."""
    current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return json.dumps({"time": current_timestamp, "timezone": timezone})

def calculate(expression):
    """Safely evaluates a basic mathematical expression string."""
    try:
        # Note: In production environments, use a safe math parser library like 'sympy' 
        # instead of eval() to avoid malicious code execution.
        result = eval(expression, {"__builtins__": None}, {})
        return json.dumps({"expression": expression, "result": str(result)})
    except Exception as e:
        return json.dumps({"error": f"Could not calculate expression: {str(e)}"})

def search_database(query, table="users"):
    """Mocks a database query lookup."""
    mock_db = {
        "users": [
            {"id": 101, "name": "Zubair", "status": "Active"},
            {"id": 102, "name": "Ayesha", "status": "Suspended"}
        ],
        "orders": [
            {"order_id": 5001, "item": "Laptop", "price": 1200},
            {"order_id": 5002, "item": "Smartphone", "price": 600}
        ]
    }
    
    # Filter the mock data list based on your query match
    records = mock_db.get(table, [])
    filtered_results = [
        r for r in records if any(str(query).lower() in str(val).lower() for val in r.values())
    ]
    
    return json.dumps({"table": table, "query": query, "results": filtered_results})

def format_currency(amount, currency="USD"):
    """Formats raw floats or integers into standardized financial currency strings."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "PKR": "Rs."}
    symbol = symbols.get(currency, "$")
    try:
        formatted = f"{symbol}{float(amount):,.2f}"
        return json.dumps({"formatted_amount": formatted, "currency": currency})
    except ValueError:
        return json.dumps({"error": "Invalid numerical amount passed."})


# =====================================================================
# STEP 2: DEFINE THE SPECIFICATION (The Blueprint Toolbox array)
# =====================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Use this tool to find out the current time and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The target timezone name, e.g., 'UTC', 'EST', 'PKT'."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Solves math equations or number calculations. Do not guess math answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math equation to solve, e.g., '24 * 56' or '45000 / 1.15'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the internal core database for matches on users or orders data records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword, name, ID, or item title to look up."
                    },
                    "table": {
                        "type": "string",
                        "enum": ["users", "orders"],
                        "description": "The secure target database table data requested."
                    }
                },
                "required": ["query", "table"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_currency",
            "description": "Formats raw input numbers into clean financial price format displaying currency symbols.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The raw numerical financial price value to format nicely."
                    },
                    "currency": {
                        "type": "string",
                        "enum": ["USD", "EUR", "GBP", "PKR"],
                        "description": "The target currency three-letter designator standard."
                    }
                },
                "required": ["amount", "currency"]
            }
        }
    }
]

# Map string tool names directly to our executable python memory objects
available_functions = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_database": search_database,
    "format_currency": format_currency
}


# =====================================================================
# STEP 3: RUN THE INTERACTIVE EXECUTION LOOP
# =====================================================================

def run_agent_loop(user_prompt):
    print(f"\n[User Query]: {user_prompt}")
    
    # Maintain context window state history array
    messages = [{"role": "user", "content": user_prompt}]
    max_turns = 5
    turn = 0
    
    while turn < max_turns:
        turn += 1
        print(f"\n--- [AGENT TURN {turn}] ---")
        # Loop Turn 1: Hand user context along with tools array to Model
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto" 
            # tool_choice = :"none"      # The model intelligently decides if tools are required
        )
    
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
    
        # Check if the Model explicitly picked/requested any function call
        if tool_calls:
            print(f"-> [Model Decision]: Tool needed! Selected function: '{tool_calls[0].function.name}'")
        
        # Append the assistant's instruction block token state to history
            messages.append(response_message)
        
        # Parse and execute each tool call request locally
            for tool_call in tool_calls:
               function_name = tool_call.function.name
               function_to_call = available_functions[function_name]
               function_args = json.loads(tool_call.function.arguments)
            
               print(f"-> [Local Execution]: Running '{function_name}' with args: {function_args}")
            
            # Execute local operational logic
               tool_output = function_to_call(**function_args)
            
            # Append execution outcome block back to message array history state
               messages.append({
                  "tool_call_id": tool_call.id,
                   "role": "tool",
                    "name": function_name,
                    "content": tool_output
                })
            # Keep looping! Go back to the top of the while block to show results to the AI
            continue
            
        # Scenario B: The model does not need tools anymore. It has the final answer.
        else:
            print("-> AI decided: No more tools needed. Synthesizing final answer.")
            return response_message.content

    return "Agent Error: Maximum operational steps exceeded without reaching a conclusion."
    #     # Loop Turn 2: Send history updated with local real data back to Model
    #     print("-> [Synthesis Phase]: Sending data back to model for final answer...")
    #     second_response = client.chat.completions.create(
    #         model="llama-3.3-70b-versatile",
    #         messages=messages
    #     )
    #     return second_response.choices[0].message.content
    # else:
    #     # Standard conversation trajectory case
    #     print("-> [Model Decision]: No functional tool needed. Standard chat reply text generation.")
    #     return response_message.content


# =====================================================================
# DEMONSTRATION RUNS
# =====================================================================
if __name__ == "__main__":
    # Test case 1: Multi-variable tool match
    # ans1 = run_agent_loop("Can you search the users database table for anyone named Zubair?")
    # print(f"[Final System Answer]: {ans1}")
    
    # print("-" * 60)
    
    # # Test case 2: Pure mathematical execution
    # ans2 = run_agent_loop("What is 4523 times 89?")
    # print(f"[Final System Answer]: {ans2}")
    complex_prompt = (
        "Look up the price of the 'Laptop' inside the orders table, "
        "calculate what a 15% sales tax on that amount would be, "
        "and format that tax value nicely into USD currency."
    )
    
    final_output = run_agent_loop(complex_prompt)
    print(f"\n[Final Agent Answer]:\n{final_output}")
