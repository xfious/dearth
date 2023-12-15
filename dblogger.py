import lmdb
import os
import pickle

import logging

class DB_logger():
    '''
    log to lmdb
    '''

    def __init__(self, logger_name, log_dir, allow_create_new_db=True):
        self.logger_name = logger_name + ".log.lmdb"
        self.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.db_path = os.path.join(log_dir, self.logger_name)
        self.db = None
        if not allow_create_new_db and not os.path.exists(self.db_path):
            raise Exception(f"db not found: {self.logger_name}")

        if not os.path.exists(self.db_path):
            logging.info(f"create new lmdb: {self.logger_name}")
            self.db = lmdb.open(self.db_path, map_size=1024*1024*1024*1024)
        else:
            self.db = lmdb.open(self.db_path, map_size=1024*1024*1024*1024)
    
    def raw_key_name_to_tuple(self, raw_key_name):
        key = raw_key_name.split("<v>")
        if len(key) == 2:
            return (None, key[0], int(key[1]))
        else:
            return (key[0], key[1], int(key[2]))


    def log(self, key, value, incremental_id, category=None):
        key = tuple_to_key_name((category, key, incremental_id))
        value = pickle.dumps(value)
        with self.db.begin(write=True) as txn:
            txn.put(key.encode('utf-8'), value)

    def get(self, key, incremental_id, category=None):
        key = tuple_to_key_name((category, key, incremental_id))
        with self.db.begin(write=False) as txn:
            value = txn.get(key.encode('utf-8'))
            if value is None:
                return None
            value = pickle.loads(value)
            return value
    
    def get_all_keys(self):
        with self.db.begin(write=False) as txn:
            keys = [key.decode('utf-8') for key, _ in txn.cursor()]
            return keys
    
    def get_all_keys_as_tuple(self):
        ret = []
        with self.db.begin(write=False) as txn:
            for key, _ in txn.cursor():
                key = key.decode('utf-8')
                key = key.split("<v>")
                if len(key) == 2:
                    ret.append((None, key[0], int(key[1])))
                else:
                    ret.append((key[0], key[1], int(key[2])))
        return ret

    def get_with_tupled_key(self, tupled_key):
        category, key, incremental_id = tupled_key
        return self.get(key, incremental_id, category)

    def get_with_tupled_key_list(self, tupled_key_list):
        '''
        tupled_key_list: [(category, key, incremental_id), ...]
        return: [value, ...]
        '''
        ret = []
        with self.db.begin(write=False) as txn:
            for tupled_key in tupled_key_list:
                category, key, incremental_id = tupled_key
                key = tuple_to_key_name((category, key, incremental_id))
                value = txn.get(key.encode('utf-8'))
                if value is None:
                    ret.append(None)
                else:
                    value = pickle.loads(value)
                    ret.append(value)
        return ret

    def get_with_mainKeyName(self, mainKeyName):
        k = []
        v = []
        with self.db.begin(write=False) as txn:
            for key, value in txn.cursor():
                key = key.decode('utf-8')
                key = raw_key_name_to_tuple(key)
                if key[1] == mainKeyName:
                    k.append(key)
                    v.append(pickle.loads(value))
        return (k, v)
    
    def __del__(self):
        if self.db is not None:
            self.db.close()
            self.db = None
    
    def copy_from(self, other_db_path):
        other_db = lmdb.open(other_db_path, map_size=1024*1024*1024*1024)
        with self.db.begin(write=True) as txn:
            with other_db.begin(write=False) as other_txn:
                for key, value in other_txn.cursor():
                    txn.put(key, value)





def raw_key_name_to_tuple(raw_key_name):
    key = raw_key_name.split("<v>")
    if len(key) == 2:
        return (None, key[0], int(key[1]))
    else:
        return (key[0], key[1], int(key[2]))
    

def tuple_to_key_name(tupled_key):
    category, key, incremental_id = tupled_key
    if category is not None:
        return f"{category}<v>{key}<v>{incremental_id}"
    else:
        return f"{key}<v>{incremental_id}"