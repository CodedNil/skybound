use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::{env, fs::OpenOptions};

// ---------- Request Payload ----------
// Represents the main structure for the API request payload.
#[derive(Deserialize, Serialize)]
struct RequestBody {
    model: String,
    max_tokens: usize,
    messages: Vec<Message>,
}

// Represents the main structure for the API request payload with funcs.
#[derive(Deserialize, Serialize)]
struct RequestBodyFuncs {
    model: String,
    max_tokens: usize,
    messages: Vec<Message>,
    functions: Vec<Value>,
}

// Represents individual messages in the request.
#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

// Represents a func description.
#[derive(Deserialize, Serialize)]
pub struct Func {
    name: String,
    description: String,
    parameters: Vec<FuncParam>,
}

impl Func {
    pub fn new(name: &str, description: &str, parameters: Vec<FuncParam>) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        }
    }
}

// Represents the params for a func.
#[derive(Deserialize, Serialize)]
pub struct FuncParam {
    name: String,
    description: String,
    required: bool,
}

impl FuncParam {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
        }
    }

    pub fn not_required(mut self) -> Self {
        self.required = false;
        self
    }
}

// ---------- API Response ----------
// Represents the expected response format from the API.
#[derive(Deserialize, Serialize)]
struct ApiResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

// Represents individual messages in the request.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MessageResponse {
    content: Option<String>,
    func_call: Option<FuncCall>,
}

// Represents individual choices in the API response.
#[derive(Deserialize, Serialize)]
struct Choice {
    message: MessageResponse,
}

// Represents the token usage of a response.
#[derive(Deserialize, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// Represents the func call with arguments.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FuncCall {
    name: String,
    arguments: Value,
}

pub fn query_openai(
    prompt: &String,
    max_tokens: usize,
    funcs: &Option<Vec<Func>>,
) -> Result<MessageResponse> {
    // Retrieve the API key from the environment variable.
    dotenv::dotenv().ok();
    let api_key = env::var("OPENAI_API_KEY").context("No OPENAI_API_KEY found in environment")?;

    let messages: Result<Vec<Message>> = prompt
        .split('|')
        .map(|message| {
            let parts: Vec<&str> = message.split(':').collect();
            let role: String = match parts.first() {
                Some(&"u") => Ok("user".into()),
                Some(&"s") => Ok("system".into()),
                Some(&"a") => Ok("assistant".into()),
                _ => Err(anyhow::anyhow!("Invalid role")),
            }?;
            let content: String = match parts.get(1) {
                Some(&content) => Ok(content.into()),
                _ => Err(anyhow::anyhow!("Invalid content")),
            }?;
            Ok(Message { role, content })
        })
        .collect();
    let messages = messages.context("Failed to parse messages")?;

    // Construct the request payload.
    let body_str = if let Some(funcs) = funcs {
        let body = RequestBodyFuncs {
            model: "gpt-3.5-turbo".into(),
            max_tokens,
            messages,
            functions: convert_funcs(funcs),
        };
        serde_json::to_string(&body).context("Failed to serialize the request body")?
    } else {
        let body = RequestBody {
            model: "gpt-3.5-turbo".into(),
            max_tokens,
            messages,
        };
        serde_json::to_string(&body).context("Failed to serialize the request body")?
    };

    // Execute the HTTP POST request to the OpenAI API.
    let raw_response = ureq::post("https://api.openai.com/v1/chat/completions")
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {api_key}"))
        .send_string(&body_str)
        .context("Failed to make the HTTP request")?;

    // Deserialize the response into our ApiResponse struct.
    let response: ApiResponse = serde_json::from_str(
        &raw_response
            .into_string()
            .context("Failed to convert the response into a string")?,
    )
    .context("Failed to parse response into JSON")?;

    // Check if there's a choice in the response and extract the assistant's reply.
    if let Some(choice) = response.choices.first() {
        // Log the required details to a log file.
        log_details(prompt, &choice.message, &response.usage)?;

        return Ok(choice.message.clone());
    }
    Err(anyhow::anyhow!(
        "Failed to extract message content from the response"
    ))
}

fn convert_funcs(funcs: &[Func]) -> Vec<Value> {
    funcs
        .iter()
        .map(|func| {
            serde_json::json!({
                "name": func.name,
                "description": func.description,
                "parameters": {
                    "type": "object",
                    "properties": func.parameters.iter().map(|param| {
                        (param.name.clone(), serde_json::json!({
                            "description": param.description,
                            "required": param.required,
                        }))
                    }).collect::<serde_json::Map<String, Value>>()
                },
                "required": func.parameters.iter()
                    .filter(|param| param.required)
                    .map(|param| param.name.clone())
                    .collect::<Vec<String>>(),
            })
        })
        .collect::<Vec<Value>>()
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn log_details(prompt: &String, result: &MessageResponse, tokens: &Usage) -> Result<()> {
    // Pricing is input $0.0015 / 1K tokens output $0.002 / 1K tokens
    let price = ((tokens.prompt_tokens as f32 * 0.0015)
        + (tokens.completion_tokens as f32 * 0.002))
        / 1000.0;

    // Format the log entry.
    let result = format!("{result:?}");
    let log_entry = format!(
        "Prompt: {:} | Result: {:} | Tokens: {}/{}/{} ${}\n",
        &prompt[..100.min(prompt.len())],
        &result[..100.min(result.len())],
        tokens.prompt_tokens,
        tokens.completion_tokens,
        tokens.total_tokens,
        price
    );

    // Open the log file in append mode.
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("gpt_log.txt")
        .context("Failed to open log file")?;

    // Write the log entry to the file.
    file.write_all(log_entry.as_bytes())
        .context("Failed to write to log file")?;

    Ok(())
}
