extends Node3D

# Variables
var Food = preload("res://Food.tscn")
var spawn_interval = 1.0


# Called when the node enters the scene tree for the first time.
func _ready():
	pass  # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
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
