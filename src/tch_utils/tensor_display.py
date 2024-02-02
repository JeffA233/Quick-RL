import gdb

class TensorDisplay:

    def __init__(self, val):
        self.val = val
        self.size = list(val['size'])
        self.kind = val['kind']
        self.max = float(val['max'])
        self.min = float(val['min'])
        self.device = val['device']

    def to_string(self):
        return "size: {}, max: {}, min: {}".format(self.size, self.max, self.min)

def lookup(val):
    lookup_tag = val.type.tag
    if lookup_tag is None:
        return None
    if "tch_ppo::utils::dbg_funcs::ten_display::TensorDisplay" == lookup_tag:
        return TensorDisplay(val)

    return None

gdb.current_objfile().pretty_printers.append(lookup)
