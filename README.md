# toy_physics

Generate a dataset of "toys" colliding with each other.

## Requirements

- TDW
- Python 3.6+
- [tdw_physics](https://github.com/alters-mit/tdw_physics)

## Usage

```bashh
python3 physics_dataset.py [ARGUMENTS]
```

| Argument | Type | Default            | Description                              |
| -------- | ---- | ------------------ | ---------------------------------------- |
| `--dir`  | str  | D:/physics_dataset | Root output directory.                   |
| `--temp` | str  | D:/temp.hdf5       | Temp file location for incomplete files. |
| `--num`  | int  | 3000               | Number of trials.                        |

For information re: the API or output data structure, see [tdw_physics](https://github.com/alters-mit/tdw_physics) (RigidbodiesDataset).