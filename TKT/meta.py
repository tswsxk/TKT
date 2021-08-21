# coding: utf-8
# 2021/8/18 @ tongshiwei

from EduKTM import KTM as _KTM


class KTM(_KTM):
    @classmethod
    def from_pretrained(cls, model_dir, *args, **kwargs):
        raise NotImplementedError
