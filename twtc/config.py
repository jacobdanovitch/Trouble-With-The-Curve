import neptune as nt
from dotenv import load_dotenv

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for k, v in kwargs.items():
            self.set(k, v)

        if self.track:
            from pathlib import Path  # python3 only
            env_path = Path('.') / '.env'
            load_dotenv(dotenv_path=env_path, verbose=True)

            nt.init(project_qualified_name='jacobdanovitch/Trouble-with-the-Curve')
            self.exp = nt.create_experiment(
                name='twtc-classifier',
                description='classifying if players will make the mlb from their scouting reports',
                params=kwargs,
                tags=self.tags
            )

        
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
        #self.parameter(key, val)

    def parameter(self, key, val):
        if hasattr(self, 'exp'):
            self.exp.set_property(key, str(val))
        return val

    def log(self, key, val):
        if hasattr(self, 'exp'):
            self.exp.send_metric(key, val)