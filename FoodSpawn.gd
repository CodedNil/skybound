extends Node3D

# Variables
var gpt = preload("res://GPT.gd").new()

var Food = preload("res://Food.tscn")
var spawn_interval = 1.0


# Called when the node enters the scene tree for the first time.
func _ready():
	var start_time = Time.get_ticks_msec()
	OS.delay_msec(1000)
	print("Before async call!", " ", Time.get_ticks_msec() - start_time)
	gpt.call_GPT("Whats 2+2?")
	print("After async call!", " ", Time.get_ticks_msec() - start_time)


func _process(delta):
	spawn_interval -= delta
	if spawn_interval <= 0:
		spawn_food_box()
		spawn_interval = 1.0


# Function to spawn the food box at a random location
func spawn_food_box():
	var food_instance = Food.instantiate()
	add_child(food_instance)
	food_instance.position = Vector3(
		(randf() - 0.5) * 20, (randf() - 0.5) * 20, (randf() - 0.5) * 20
	)
