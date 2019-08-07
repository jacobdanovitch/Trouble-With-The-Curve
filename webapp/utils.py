def clean_text(corp, rm_null=True):
    if rm_null:
        corp = corp[~corp.isnull()]
    corp = corp.str.lower().str.encode('ascii', 'ignore').str.decode('ascii')
    corp = corp.str.replace("bernie pleskoff's scouting report", "")
    corp = corp.apply(lambda doc: ''.join([char if char.isalpha() else f' {char} ' for char in doc]))
    return corp