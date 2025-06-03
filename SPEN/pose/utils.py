from .discrete_euler import DiscreteEulerEncoder, DiscreteEulerDecoder
from .quat import QuatEncoder, QuatDecoder
from .discrete_spher import DiscreteSpherEncoder, DiscreteSpherDecoder
from .cart import CartEncoder, CartDecoder
from .euler import EulerEncoder, EulerDecoder
from .spher import SpherEncoder, SpherDecoder

def get_ori_encoder(ori_type: str, **kwargs):
    if ori_type == "DiscreteEuler":
        return DiscreteEulerEncoder(**kwargs)
    elif ori_type == "Quat":
        return QuatEncoder(**kwargs)
    elif ori_type == "Euler":
        return EulerEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown ori type: {ori_type}")

def get_ori_decoder(ori_type: str, **kwargs):
    if ori_type == "DiscreteEuler":
        return DiscreteEulerDecoder(**kwargs)
    elif ori_type == "Quat":
        return QuatDecoder(**kwargs)
    elif ori_type == "Euler":
        return EulerDecoder(**kwargs)
    else:
        raise ValueError(f"Unknown ori type: {ori_type}")


def get_pos_encoder(pos_type: str, **kwargs):
    if pos_type == "DiscreteSpher":
        return DiscreteSpherEncoder(**kwargs)
    elif pos_type == "Cart":
        return CartEncoder(**kwargs)
    elif pos_type == "Spher":
        return SpherEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown pos type: {pos_type}")


def get_pos_decoder(pos_type: str, **kwargs):
    if pos_type == "DiscreteSpher":
        return DiscreteSpherDecoder(**kwargs)
    elif pos_type == "Cart":
        return CartDecoder(**kwargs)
    elif pos_type == "Spher":
        return SpherDecoder(**kwargs)
    else:
        raise ValueError(f"Unknown pos type: {pos_type}")