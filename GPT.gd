extends Node

const OPENAI_API_KEY: String = "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"
var http_request: HTTPRequest


func _ready():
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.connect("request_completed", _on_request_completed)


func call_GPT(prompt):
	var body = JSON.new().stringify(
		{
			"messages": [{"role": "user", "content": prompt}],
			"temperature": 0.7,
			"max_tokens": 1024,
			"model": "gpt-3.5-turbo"
		}
	)
	var error = http_request.request(
		"https://api.openai.com/v1/chat/completions",
		["Content-Type: application/json", "Authorization: Bearer " + OPENAI_API_KEY],
		HTTPClient.METHOD_POST,
		body
	)

	if error != OK:
		push_error("Something Went Wrong!")


func _on_request_completed(result, responseCode, headers, body):
	printt(result, responseCode, headers, body)
	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	if parse_result:
		printerr(parse_result)
		return
	var response = json.get_data()
	if response is Dictionary:
		printt("Response", response)
		if response.has("error"):
			printt("Error", response["error"])
			return
	else:
		printt("Response is not a Dictionary", headers)
		return

	var newStr = response.choices[0].message.content
	printt("New String ", newStr)
	pass
