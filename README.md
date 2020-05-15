# tdw_physics

Generate a dataset of "toys" colliding with each other.

## Usage

```bashh
python3 physics_dataset.py [ARGUMENTS]
```

| Argument | Type | Default            | Description            |
| -------- | ---- | ------------------ | ---------------------- |
| `--dir`  | str  | D:/physics_dataset | Root output directory. |
| `--num`  | int  | 3000               | Number of trials.      |

## Output

Each trial outputs a separate .hdf5 file in the root output directory. The files are named sequentially, e.g.:

```
root/
....0000.hdf5
....0001.hdf1
```

#### File Structure

```
static/    # Data that doesn't change per frame.
....object_ids
....mass
....static_friction
....dynamic_friction
....bounciness
frames/    # Per-frame data.
....0000/    # The frame number.
........images/    # Each image pass.
............_img
............_id
............_depth
............_normals
........objects/    # Per-object data.
............positions
............forwards
............rotations
............velocities
............angular_velocities
........collisions/    # Collisions between two objects.
............object_ids
............relative_velocities
............contacts
........env_collisions/    # Collisions between one object and the environment.
............object_ids
............contacts
....0001/
........ (etc.)
```

- All object data is ordered to match `object_ids`. For example:
  - `static/mass[0]` is the mass of `static/object_ids[0]`
  - `frames/0000/positions[0]` is the position of `static/object_ids[0]`
- The shape of each dataset in `objects` is determined by the number of coordinates. For example, `frames/objects/positions/` has shape `(num_objects, 3)`.
- The  shape of all datasets in `collisions/` and `env_collisions/`are defined by the number of collisions on that frame.
  -  `frames/collisions/relative_velocities` has the shape `(num_collisions, 3)`
  - `frames/collisions/object_ids` has the shape `(num_collisions, 2)` (tuple of IDs).
  - `frames/env_collisions/object_ids` has the shape `(num_collisions)` (only 1 ID per collision).
  - `frames/collisions/contacts` and `frames/env_collision/contacts` are tuples of `(normal, point)`, i.e. the shape is `(num_collisions, 2, 3)`.