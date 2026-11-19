from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    path: str
    height: int
    width: int
    pool_id: str
    sha256: str
    optional_case_id: str | None = None


@dataclass(frozen=True)
class EvalAnnotation:
    sample_id: str
    y: int
    mask_path: str | None
    annotation_kind: str = 'binary_anomaly_mask'


def load_rows(path):
    with open(path,encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def image_record(row):
    return ImageRecord(row['sample_id'], row['path'], row['height'], row['width'],
                       row['category_id'], row['image_sha256'],row.get('optional_case_id'))


def save_rows(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
