from copy import deepcopy

# Find string between specific strings
def find_string_between(s, start, end=None):
    s_backup = deepcopy(s)
    s = s.lower()
    start = start.lower()
    if end is not None:
        end = end.lower()
    start_idx = s.find(start)
    if start_idx == -1:
        return None
    start_idx += len(start)
    if end is None:
        return s_backup[start_idx:].strip()
    else:
        end_idx = s.find(end, start_idx)
        if end_idx == -1:
            return s_backup[start_idx:].strip()
        else:
            return s_backup[start_idx:end_idx].strip()