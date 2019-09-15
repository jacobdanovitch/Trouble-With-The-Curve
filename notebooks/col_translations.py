am_bat = {
    'age': 'Age', 
    # 'teamId', 
    # 'full_name', 
    # 'playerId': 'minormasterid', 
    'atBats': 'AB',
    'runs': 'R', 
    'hits': 'H', 
    'doubles': '2B', 
    'triples': '3B', 
    'homeRuns': 'HR', 
    'obp': 'OBP', 
    'ops': 'OPS', 
    'slg': 'SLG',
    'rbi': 'RBI', 
    'baseOnBalls': 'BB', 
    'strikeOuts': 'SO', 
    'stolenBases': 'SB', 
    'caughtStealing': 'CS',
    # 'leftOnBase', 
    # 'totalBases', 
    'avg': 'AVG', 
    # 'position',
    'sportAbbrev': 'Level'
}

am_pit = {
    # 'rank', 
    # 'team', 
    'age': 'Age', 
    # 'teamId', 
    # 'playerId', 
    # 'full_name', 
    'gamesPitched': 'GP', 
    'whip': 'WHIP', 
    'inningsPitched': 'IP',
    'baseOnBalls': 'BB', 
    'strikeOuts': 'SO',
    'homeRuns': 'HR',
    # 'H', 
    # 'RBI', 
    # 'earnedRuns',
    # 'BB',
    # 'SO', 
    # 'HR', 
    # 'outs', 
    'battersFaced': 'TBF', 
    'pitchesThrown': 'Pitches', 
    'era': 'ERA',
    'saves': 'SV', 
    'holds': 'Hld', 
    'blownSaves': 'BS', 
    'wins': 'W', 
    'losses': 'L', 
    #'Level', 
    #'Season'
}


api_hit = {
    'hr': 'HR',
    'rbi': 'RBI',
    'bb': 'BB',
    'avg': 'AVG',
    'slg': 'SLG',
    'ops': 'OPS',
    'so': 'SO',
    'h': 'H' ,
    'cs': 'CS',
    'obp': 'OBP',
    'r': 'R',
    'sb': 'SB',
    'ab': 'AB',
    'sport': 'Level'
}

api_pit = {
    'hr': 'HR_pit',
    'era': 'ERA',
    'sv': 'SV',
    # 'avg': 'AVG,
    'whip': 'WHIP',
    'bb': 'BB_pit',
    'so': 'SO_pit',
    'tbf': 'TBF',
    'l': 'L',
    # 'h': 'H',
    'ip': 'IP',
    'w': 'W',
    'sport': 'Level'
    # 'r',
    # 'ab'
}


am_col_map = {
    'batters': am_bat,
    'pitchers': am_pit
}

api_col_map = {
    'hitting': api_hit,
    'pitching': api_pit
}