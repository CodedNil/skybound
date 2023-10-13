use anyhow::{Context, Result};
use godot::prelude::godot_print;
use serde_derive::{Deserialize, Serialize};
use std::env;

// Struct for the message format in both request and response.
#[derive(Serialize, Deserialize)]
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

pub fn query_openai(prompt: String) -> Result<String> {
    // Retrieve the API key from the environment variable.
    let api_key = env::var("OPENAI_API_KEY").context("No OPENAI_API_KEY found in environment")?;

    // Construct the request payload.
    let body = RequestBody {
        model: "gpt-3.5-turbo".into(),
        max_tokens: 1024,
        messages: vec![
            Message {
                role: "system".into(),
                content: "You are a helpful assistant.".into(),
            },
            Message {
                role: "user".into(),
                content: prompt,
            },
        ],
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

    // Print token usage.
    godot_print!("Prompt tokens: {}", response.usage.prompt_tokens);
    godot_print!("Completion tokens: {}", response.usage.completion_tokens);
    godot_print!("Total tokens: {}", response.usage.total_tokens);

    // Check if there's a choice in the response and extract the assistant's reply.
    if let Some(choice) = response.choices.first() {
        return Ok(choice.message.content.clone());
    }
    Err(anyhow::anyhow!(
        "Failed to extract message content from the response"
    ))
}
