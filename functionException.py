import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

# Load secure API keys from a local .env file
load_dotenv()

# Initialize the Groq cloud pipeline client
client = Groq()
# =====================================================================
# STEP 1: ROBUST LOCAL FUNCTIONS (With Strict Type Validation & Capture)
# =====================================================================

def get_current_time(timezone="UTC"):
    """Returns the current system date and time safely."""
    try:
        current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return json.dumps({"status": "success", "time": current_timestamp, "timezone": timezone})
    except Exception as e:
        # Handles any unexpected system clock exceptions safely [1]
        return json.dumps({"status": "error", "message": f"System clock failure: {str(e)}"})

def calculate(expression):
    """Safely evaluates math expressions and captures calculation errors [1]."""
    try:
        # Basic check to avoid execution of malicious code injections
        if any(char.isalpha() for char in expression if char not in ['e', 'pi']):
            return json.dumps({"status": "error", "message": "Security Reject: Letters are forbidden inside math expressions."})
            
        result = eval(expression, {"__builtins__": None}, {})
        return json.dumps({"status": "success", "expression": expression, "result": str(result)})
    except ZeroDivisionError:
        # Handles division by zero functional exceptions cleanly [1]
        return json.dumps({"status": "error", "message": "Math Error: Cannot divide a number by zero."})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Syntax Error: Invalid math arrangement ({str(e)})"})

def search_database(query, table="users"):
    """Mocks a database query lookup with validation safety layers [1]."""
    allowed_tables = ["users", "orders"]
    if table not in allowed_tables:
        return json.dumps({"status": "error", "message": f"Access Denied: Table '{table}' does not exist or is protected."})

    mock_db = {
        "users": [
            {"id": 101, "name": "Zubair", "status": "Active"},
            {"id": 102, "name": "Ayesha", "status": "Suspended"}
        ],
        "orders": [
            {"order_id": 5001, "item": "Laptop", "price": 1200.00},
            {"order_id": 5002, "item": "Smartphone", "price": 600.00}
        ]
    }
    
    try:
        records = mock_db.get(table, [])
        filtered_results = [
            r for r in records if any(str(query).lower() in str(val).lower() for val in r.values())
        ]
        return json.dumps({"status": "success", "table": table, "query": query, "results": filtered_results})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Database lookup failure: {str(e)}"})

def format_currency(amount, currency="USD"):
    """Handles Scenario 2: Explicitly intercepts invalid argument type errors [1]."""
    try:
        # Force convert data type to float at runtime to check validity
        numeric_amount = float(amount)
    except (ValueError, TypeError):
        # Catching strings like "one-hundred" or dictionary mismatch bugs [1]
        return json.dumps({
            "status": "error", 
            "message": f"Data Type Error: The system expected an integer or float, but the AI passed a value with type '{type(amount).__name__}' containing content '{amount}'."
        })

    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "PKR": "Rs."}
    symbol = symbols.get(currency, "$")
    
    try:
        formatted = f"{symbol}{numeric_amount:,.2f}"
        return json.dumps({"status": "success", "formatted_amount": formatted, "currency": currency})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Formatting execution anomaly: {str(e)}"})
# =====================================================================
# STEP 2: TOOLS BLUEPRINT SPECIFICATION WITH STRICT WARNINGS
# =====================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Find out the current time and date.",
            "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "CRITICAL: You MUST use this tool for ANY math or calculations. Do not guess answers using standard text generation.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "The math string to calculate"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the core database for matches on users or orders data records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term"},
                    "table": {"type": "string", "enum": ["users", "orders"]}
                },
                "required": ["query", "table"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_currency",
            "description": "Converts raw numeric numbers into financial price formats with symbols. Amount parameter MUST be digits only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "The precise numeric value, e.g., 150.50"},
                    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "PKR"]}
                },
                "required": ["amount", "currency"]
            }
        }
    }
]

available_functions = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_database": search_database,
    "format_currency": format_currency
}
# =====================================================================
# STEP 3: RESILIENT EXECUTION CORE ENGINE
# =====================================================================

def run_secure_agent(user_prompt):
    print(f"\n[User Query]: {user_prompt}")
    messages = [{"role": "user", "content": user_prompt}]
    
    max_turns = 5
    turn = 0
    
    while turn < max_turns:
        turn += 1
        print(f"--- [Loop Turn {turn}] ---")
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as api_err:
            return f"Network Crash: Unable to contact Groq cloud cluster. Details: {str(api_err)}"
            
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # Action Block A: Tool call is generated by AI
        if tool_calls:
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                
                # Check for nonexistent hallucinated tools
                if function_name not in available_functions:
                    print(f"-> [Error Handled] Hallucinated Tool name checked: '{function_name}'")
                    error_payload = {"status": "error", "message": f"Tool '{function_name}' is locked or unavailable."}
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(error_payload)
                    })
                    continue
                
                function_to_call = available_functions[function_name]
                
                # Catch bad JSON formatting structures directly from AI stream
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    print("-> [Error Handled] Malformed JSON arguments string captured.")
                    error_payload = {"status": "error", "message": "Argument structural compilation error. Ensure arguments are pure JSON format."}
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(error_payload)
                    })
                    continue
                
                print(f"-> AI Requested: '{function_name}' with args: {function_args}")
                
                # Run function. Internal try/except blocks handle bad datatypes or runtime math exceptions
                tool_output = function_to_call(**function_args)
                print(f"-> Secure Return: {tool_output}")
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_output
                })
            
            continue # Re-submit loop update to let AI inspect success/error responses
            
        # Action Block B: No tool call generated by AI
        else:
            # Handles Scenario 1: AI tries to bypass tool and guess calculations
            if any(word in user_prompt.lower() for word in ["times", "divided", "+", "*", "minus", "multiplied"]):
                if len(messages) == 1: # Intercept only on the initial step
                    print("-> [Guardrail Activated]: Model tried to bypass the math tool and guess. Forcing tool alignment...")
                    
                    # Force the model to generate arguments for the calculate function
                    forced_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": "calculate"}}
                    )
                    response_message = forced_response.choices[0].message
                    tool_calls = response_message.tool_calls
                    if tool_calls:
                        continue
            
            print("-> Response finalized safely.")
            return response_message.content

    return "System Error: Safety operations execution threshold reached without stable closeout."


# =====================================================================
# SYSTEM EVALUATION TRAPS
# =====================================================================
if __name__ == "__main__":
    
    print("=" * 70)
    print("EVALUATION 1: FORCING A MATH OVERRIDE (Scenario 1)")
    print("=" * 70)
    output1 = run_secure_agent("Calculate 4523 divide by 0  inside your head right now.")
    print(f"\n[Final Agent Answer]:\n{output1}\n")
    
    print("=" * 70)
    print("EVALUATION 2: INVALID DATA TYPE HANDLING (Scenario 2)")
    print("=" * 70)
    output2 = run_secure_agent("Format the text string string 'one-thousand-dollars' into USD currency format.")
    print(f"\n[Final Agent Answer]:\n{output2}\n")
