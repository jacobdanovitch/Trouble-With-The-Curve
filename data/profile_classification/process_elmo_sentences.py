import sys
import pandas as pd


def clean_text(corp, rm_null=True):
    if rm_null:
        corp = corp[~corp.isnull()]
    corp = corp.str.lower().str.encode('ascii', 'ignore').str.decode('ascii')
    corp = corp.str.replace("bernie pleskoff's scouting report", "")
    corp = corp.str.replace('\r\n', ' ')
    corp = corp.apply(lambda doc: ''.join([char if char.isalnum() else f' {char} ' if doc else None for char in doc]))
    return corp

if __name__ == '__main__':
    fp = sys.argv[1]
    read_fn = None
    
    if fp.endswith('.json'):
        read_fn = pd.read_json
    elif fp.endswith('.csv'):
        read_fn = pd.read_csv
    else:
        raise ValueError(f'Invalid file type: {fp}')
    
    df = read_fn(fp)
    text = clean_text(df['report'], rm_null=False).values

    try:
        n = int(sys.argv[2])
        text = text[:n]
    except Exception as e:
        print(e)
        pass

    print(f'Writing {len(text)} lines.')
    op = ''.join(fp.split('.')[:-1])
    with open(f'elmo/{op}.txt', 'w') as f:
        f.write('\n'.join(text.tolist()))