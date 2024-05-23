from typing import Iterator,List
import os.path


class LivnehData:
    DATA_ROOT = "./data/input/"
    RESOLUTION = 1/16
    YEARS: List[int] = list(range(1915, 2012))

    @classmethod
    def input_file(cls, year: int) -> str:
        return cls.input_file_path(f"prec.{year}.nc")

    @classmethod
    def input_files_glob(cls) -> str:
        return cls.input_file_path("prec.*.nc")

    @classmethod
    def input_file_path(cls, filename: str) -> str:
        return os.path.join(LivnehData.DATA_ROOT, filename)

    @classmethod
    def all_input_files(cls) -> Iterator[str]:
        for p in os.listdir(LivnehData.DATA_ROOT):
            if p.endswith(".nc"):
                yield cls.input_file_path(p)
