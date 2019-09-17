import pandas as pd
import numpy as np

from utils import tqdm_parallel
from col_translations import *

from tqdm.auto import tqdm # ; tqdm.pandas()
import requests

from glob import glob
import os

import re
import json

order = ['name', 'key_mlbam', 'key_fangraphs', 'key_bbref', 'key_bbref_minors', 'key_uuid', 'mlb_played_first',
       'birthdate', 'debut_age',  'age', 'year', 'primary_position', 'eta', 'report', 'Arm',
       'Changeup', 'Control', 'Curveball', 'Cutter', 'Fastball', 'Field',
       'Hit', 'Power', 'Run', 'Slider', 'Splitter', 'source']

sub_dct = {
    'ALL \(\d\)': 'UNK',
    '0': 'UNK',
    'A\(Adv\)': 'A+',
    'A\(Full\)': 'A',
    'A\(Short\)': 'A-',
    'ROA|ROK': 'R'
}


def load_twtc(fp='../data/twtc.csv'):
    twtc = pd.read_csv(fp)

    twtc.report = twtc.report.str.strip().replace('', np.nan)

    twtc = twtc.dropna(subset=['report'])

    twtc['old_mlbam'] = twtc.key_mlbam
    twtc.key_mlbam = pd.to_numeric(twtc.key_mlbam, errors='coerce').fillna(-1).astype('int')

    twtc['fg_season_id'] = twtc.key_fangraphs + "_" + twtc.year.astype(str)
    twtc['am_season_id'] = twtc.key_mlbam.astype(str) + "_" + twtc.year.astype(str)

    twtc = twtc[~twtc[['am_season_id', 'report']].duplicated()]
    twtc = twtc[~twtc[['fg_season_id', 'report']].duplicated()]
    
    twtc['primary_position'].replace('DH', '1B', inplace=True)
    twtc['primary_position'].replace('UTIL', 'INF', inplace=True)
    
    twtc = twtc.reset_index(drop=True)
    return twtc

def load_fg(players, id_list):
    fp = f'../data/fangraphs/{players}.csv'
    fg = pd.read_csv(fp)
    
    fg['Level'] = fg.Team.str.extract('\((.*)\)')
    fg['position'] = fg.Name.str.extract('position=(.*)[\/]*\"')[0].str.split('/').str[0]
    
    fg['fg_season_id'] = fg.apply(lambda x: x['minormasterid'] 
                                  if x['minormasterid'] + "_" + str(x['Season']) in id_list
                                       else x['playerids'],
                                    axis=1) + "_" + fg.Season.astype(str) # + "_" + fg.Level
    
    # fg splits by team (getting traded, promoted), so take statline with more playing time
    fg = fg.sort_values('Pitches', ascending=False) 
    fg = fg[~fg[['fg_season_id']].duplicated()] # , 'Level'
    
    # fg['uid'] = (fg.PlayerId.astype(str) + "_" + fg.Season.astype(str))
    # fg = fg.sort_values(['uid', 'Pitches'], ascending=False).reset_index(drop=True)
    # fg = fg[~fg.uid.duplicated()]
    
    # fg = fg[fg.position != 'P']

    return fg

def load_mlbam(players='batters'):
    def csv_with_year(f):
        df = pd.read_csv(f)
        df['Season'] = int(re.findall(r'\/(\d+)', f)[0])
        
        return df
    
    files = glob(f'../data/mlbam/*{players}.csv')
    df = pd.concat([csv_with_year(f) for f in files], axis=0, ignore_index=True, sort=False)
    
    cols = am_col_map[players]
    # cols = {o: f'{c}_{players[:3]}' for (o, c) in cols.items()}
    df = df.rename(columns=cols)
    df['am_season_id'] = df.playerId.astype(str) + "_" + df.Season.astype(str) # + "_" + df.Level
    
    return df


def get_kept_columns(original_cols):
    merging_cols = set(am_col_map['batters'].values())
    merging_cols.update({f'{c}_pit' if c in merging_cols else c for c in am_col_map['pitchers'].values()})

    to_keep = original_cols + list(merging_cols)
    return to_keep

    
def fill_nas(df, suffix):
    dup_cols = df.filter(regex=f'{suffix}$', axis=1).columns
    kept_cols = [c.replace(suffix, '') for c in dup_cols]
    
    assert len(kept_cols) == len(dup_cols)
    
    df = df.fillna({k: df[d] for (k, d) in zip(kept_cols, dup_cols)})
    df = df.drop(columns=dup_cols)
    return df


def fix_level(lvl):
    for sub in sub_dct:
        if bool(re.search(sub, lvl)):
            return sub_dct[sub]
    return lvl


def load_stats_df(twtc):
    to_keep = get_kept_columns(twtc.columns.tolist())
    
    id_df = twtc[['am_season_id', 'fg_season_id']].copy()
    twtc_fg_ids = twtc.key_fangraphs.unique()
    
    am_bat = load_mlbam('batters')
    fg_bat = load_fg('batters', twtc_fg_ids)
    
    am_pitch = load_mlbam('pitchers')
    fg_pitch = load_fg('pitchers', twtc_fg_ids)

    df = id_df.merge(fg_bat, how='left', 
                on='fg_season_id',
                suffixes=('', '')) # _fg_bat

    df = df.merge(am_bat, how='left', 
                on='am_season_id',  
                suffixes=('', '_am_bat'))

    df = fill_nas(df, '_am_bat')[[c for c in to_keep if c in df.columns]]
    
    df = df.merge(fg_pitch, how='left', 
            on='fg_season_id',
            suffixes=('', '_pit')) 

    df = df.merge(am_pitch, how='left', 
               on='am_season_id',
              suffixes=('', '_pit_am')
    )
    
    df = fill_nas(df, '_pit_am')[[c for c in to_keep if c in df.columns]]
    df = df.fillna({c: 0 for c in df.columns.difference(id_df.columns)})
    
    df.Level = df.Level.astype(str).apply(fix_level)#.value_counts()
    
    return df


def mlb_api_request(p, cache='../data/mlb_api'):
    pid = p['key_mlbam']
    yr = p['year']
    
    fp = f'{cache}/{pid}_{yr}.json'
    if os.path.exists(fp):
        with open(fp) as f:
            js_data = f.read()
            if js_data:
                return json.loads(js_data)
        
    if pid < 0:
        return {}
    
    pos = 'pitching' if p['primary_position'].upper().endswith('HP') else 'hitting'
    url = f"http://lookup-service-prod.bamgrid.com/lookup/json/named.sport_{pos}_composed.bam"
    params = {
        "player_id": pid,
        "game_type": "'R'",
        "league_list_id": "'mlb_milb'",
        "sort_by": "'season_asc'",
        "sport_hitting_composed.season": yr
    }
    
    r = requests.get(url, params=params)
    res = r.json()[f'sport_{pos}_composed']
    res['index'] = p['index']
    
    with open(fp, 'w') as f:
        f.write(json.dumps(res))
    
    return res
   

    
# can't curry cache var b/c mp can only pickle top-level functions
def request_missing_mlbam(ps, cache='../data/mlb_api', processes=2):
    return list(tqdm_parallel(mlb_api_request, ps, processes))
    
    
    
def parse_mlb_api_res(res):
    pos = 'hitting' if res.get('sport_career_hitting_lg') else 'pitching'
    data = res[f'sport_career_{pos}_lg']['queryResults'].get('row')
    
    if isinstance(data, list):
        for row in data:
            if row['sport'] != 'MLB':
                data = row
        #data = data[0]
    if not data:
        return {}
    
    col_map = api_col_map[pos]
    
    out = {}
    for k, v in data.items():
        if k in col_map:
            out[col_map[k]] = pd.to_numeric(v, errors='ignore')
    
    out['Level'] = fix_level(out['Level']) # data['sport']
    out['index'] = res['index']
    return out    
    

    
def fill_missing_data(df, stat_cols, processes=2):
    # df = df.loc[:,~df.columns.duplicated()]

    no_stats = (df[stat_cols].sum(axis=1) == 0)
    has_keyam = ~(df.key_mlbam.replace(-1, np.nan).isnull())
    #dup_key_yr = df[['key_mlbam']].duplicated()

    data = json.loads(df[no_stats].reset_index().to_json(orient='records')) # & has_keyam & ~dup_key_yr
    #missing = request_missing_mlbam(data, processes=processes)
    
    # allows us to curry df
    """
    def fill_missing_row(res):
        idx = res['index']
        data = parse_mlb_api_res(res)

        if not data:
            return {}

        row = df.loc[idx]
        row = row.replace(0, np.nan).replace('UNK', np.nan).fillna(data).fillna(0)
        return idx, row
    """
    
    fill_df = pd.DataFrame(df.loc[no_stats & has_keyam, ['key_mlbam', 'year']] \
                        .apply(mlb_api_request, axis=1) \
                        .apply(parse_mlb_api_res) \
                        .apply(pd.Series)
                      )
    
    #corrected_rows = dict(filter(bool, map(fill_missing_row, tqdm(missing))))
    #idxs, rows = map(list, zip(*corrected_rows.items()))
    #df.update(pd.DataFrame(rows, index=idxs))
    df.update(fill_df)

    # df.loc[rows, stat_cols] = values
    return df
    
    
