from ast import literal_eval
import pandas as pd

def clean_text(corp, rm_null=True):
    if rm_null:
        corp = corp[~corp.isnull()]
    corp = corp.str.lower().str.encode('ascii', 'ignore').str.decode('ascii')
    corp = corp.str.replace("bernie pleskoff's scouting report", "")
    corp = corp.apply(lambda doc: ''.join([char if char.isalnum() else f' {char} ' if doc else None for char in doc]))
    return corp

def parse_grades(gr):
  gr = literal_eval(gr)
  for k, v in gr.items():
    v = int(v)
    if v < 10:
      v *= 10
    gr[k] = v
  return gr

def load_df(dataset, 
            columns=['name', 'label', 'rank', 'drafted', 'eta', 'report'], #'team_file_code', 'position', 
            excluded=['O', 'FB', 'Palmball', 'Screwball', 'Change', 'Curve', 'Knuckle', 'ETA', 'Split', 'Defense', 'Speed', 'slider', 'grades'],
            root_url='https://github.com/jacobdanovitch/Trouble-With-The-Curve/blob/master/data/profile_classification/{}.json?raw=true'
           ):
  df = pd.read_json(root_url.format(dataset))
  
  df['drafted'] = df['drafted'].str.extract(r'\((\d+)\)')
  df.loc[df.drafted.isnull(), 'drafted'] = 1500 # undrafted
  df['drafted'] = df['drafted'].apply(int)
  
  df['eta'] = pd.to_numeric(df['eta'], errors='coerce')
  df = df[~df.eta.isnull()]
  df['eta'] = df['eta'] - pd.to_numeric(df['year'], errors='coerce')
  
  df['grades'] = df['grades'].apply(parse_grades)
  gr = pd.DataFrame.from_records(df['grades'], index=df.index).fillna(-1)
  gr = gr[gr.columns.difference(excluded)]

  df = df[columns]
  df = pd.concat([df, gr], axis=1, sort='True')
  
  return df