extends Node3D

const PARTICLE_DISTANCE: float = 0.25
const PARTICLE_MASS: float = PARTICLE_DISTANCE * PARTICLE_DISTANCE


class Connection:
	var target: Particle
	var direction: Vector3
	var distance: float
	var active: bool

	func _init(origin_particle: Particle, target_particle: Particle):
		self.target = target_particle
		self.direction = (origin_particle.position - target_particle.position).normalized()
		self.distance = origin_particle.position.distance_to(target_particle.position)
		self.active = true


# Define particle and connection data structures
class Particle:
	var position: Vector3
	var velocity: Vector3 = Vector3.ZERO
	var connections: Array = []
	var anchored = false

	func _init(pos: Vector3):
		self.position = pos

	func add_connection(target_particle: Particle):
		self.connections.append(Connection.new(self, target_particle))


var particles: Array = []


func _ready():
	var grid_size: int = 4

	# Create grid of particles
	for x in range(-grid_size, grid_size + 1):
		for y in range(-grid_size, grid_size + 1):
			for z in range(-grid_size, grid_size + 1):
				var particle = Particle.new(
					self.global_transform.origin + Vector3(x, y, z) * PARTICLE_DISTANCE
				)
				if (
					particle.position.x == self.global_transform.origin.x
					and self.position.y == self.global_transform.origin.y
				):
					particle.anchored = true
				particles.append(particle)

	# Connect adjacent particles based on distance threshold
	for p1 in particles:
		for p2 in particles:
			if p1 != p2 and p1.position.distance_to(p2.position) <= PARTICLE_DISTANCE * 2:
				p1.add_connection(p2)


var connections_severed = false


func _process(delta):
	update_physics(delta)

	if Time.get_ticks_msec() > 3000 and not connections_severed:
		sever_connections_on_circle(Vector3(1, -0.8, 0), 1, Vector3(0, 1, 0))
		sever_connections_on_circle(Vector3(1, -0.8, -1), 1, Vector3(0, 1, 0))
		sever_connections_on_circle(Vector3(1, -0.8, 1), 1, Vector3(0, 1, 0))
		sever_connections_on_circle(Vector3(0.75, -0.8, 0), 1, Vector3(0, 1, 0))
		sever_connections_on_circle(Vector3(0.75, -0.8, -1), 1, Vector3(0, 1, 0))
		sever_connections_on_circle(Vector3(0.75, -0.8, 1), 1, Vector3(0, 1, 0))
		connections_severed = true

	# Draw particles
	for particle in particles:
		DebugDraw3D.draw_sphere(particle.position, PARTICLE_DISTANCE * 0.25, Color(0.25, 1, 0.25))

	# Draw connections
	for particle in particles:
		for connection in particle.connections:
			if connection.active:
				var color = Color(0, 1, 0)
				if not connection.active:
					color = Color(1, 0, 0)
				var direction = (connection.target.position - particle.position).normalized()
				var line_start = particle.position + direction * connection.distance * 0.25
				var line_end = connection.target.position - direction * connection.distance * 0.25
				DebugDraw3D.draw_arrow_line(line_start, line_end, color, 0.1)


func sever_connections_on_circle(
	sever_position: Vector3, sever_radius: float, sever_normal: Vector3
):
	# Render out the circle
	var debug_transform_scaled = (
		Transform3D()
		. scaled(Vector3(sever_radius, 0, sever_radius) * 2.0)
		. rotated(Vector3(sever_normal.z, sever_normal.y, sever_normal.x), PI / 2)
		. translated(sever_position)
	)
	DebugDraw3D.draw_cylinder(debug_transform_scaled, Color(1, 0, 0), 5.0)

	# Find and sever connections
	for particle in particles:
		for connection in particle.connections:
			var p1 = particle.position
			var p2 = connection.target.position

			var t = (sever_normal.dot(sever_position - p1)) / sever_normal.dot(p2 - p1)
			if t >= 0 and t <= 1:
				var intersection_point = p1 + t * (p2 - p1)
				var intersection_dist = intersection_point.distance_to(sever_position)
				if intersection_dist < sever_radius:
					connection.active = false


var GRAVITY = Vector3(0, -5, 0) * PARTICLE_MASS
var SPRING_CONSTANT = 30.0
var REPULSION_STRENGTH = 40.0
var DAMPING = 0.95

func update_physics(delta):
	for p in particles:
		if not p.anchored:
			# Initialize net force with gravity
			var net_force = GRAVITY

			# Connection Forces
			for connection in p.connections:
				if connection.active:
					var target_position = (
						connection.target.position + connection.direction * connection.distance
					)
					var direction = target_position - p.position
					var spring_force = direction * SPRING_CONSTANT
					net_force += spring_force

			# Repulsion Forces
			for other in particles:
				if other != p:
					var difference = p.position - other.position
					var distance = difference.length()
					if distance < PARTICLE_DISTANCE:
						var repulsion_force = (
							difference.normalized()
							* (PARTICLE_DISTANCE - distance)
							* REPULSION_STRENGTH
						)
						net_force += repulsion_force

			# Update velocity and position
			p.velocity += net_force * delta
			p.velocity *= DAMPING
			p.position += p.velocity * delta
