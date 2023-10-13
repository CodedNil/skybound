use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::{env, fs::OpenOptions};

// Struct for the message format in both request and response.
#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

// Struct for formatting the API request body.
#[derive(Serialize)]
struct RequestBody {
    model: String,
    max_tokens: usize,
    messages: Vec<Message>,
}

// Structs for parsing the API response.
#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Deserialize)]
struct ApiResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

pub fn query_openai(prompt: &String) -> Result<String> {
    // Retrieve the API key from the environment variable.
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
    let body = RequestBody {
        model: "gpt-3.5-turbo".into(),
        max_tokens: 256,
        messages,
    };

    // Serialize the request payload to a JSON string.
    let body_str = serde_json::to_string(&body).context("Failed to serialize the request body")?;

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
        log_details(prompt, &choice.message.content, &response.usage)?;

        return Ok(choice.message.content.clone());
    }
    Err(anyhow::anyhow!(
        "Failed to extract message content from the response"
    ))
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn log_details(prompt: &String, result: &String, tokens: &Usage) -> Result<()> {
    // Pricing is input $0.0015 / 1K tokens output $0.002 / 1K tokens
    let price = ((tokens.prompt_tokens as f32 * 0.0015)
        + (tokens.completion_tokens as f32 * 0.002))
        / 1000.0;
    // Format the log entry.
    let log_entry = format!(
        "Prompt: {:<10} | Result: {:<10} | Tokens: {}|{}/{} ${}\n",
        &prompt[..10.min(prompt.len())],
        &result[..10.min(result.len())],
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
