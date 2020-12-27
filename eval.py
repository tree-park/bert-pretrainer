
from transformer_lm.lib.util import Config
from lib.bert_pretrain import BERTEmbedding

# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/lm.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

# load w2v model and train
lm = BERTEmbedding(dconf, mconf)
lm.load('trained.pth')

test = ['또 하나 필요한 것은 훌륭한 영어 실력이다.', '또 하나 필요한 것은 훌륭한 영어 실력이다.', '경찰은 월요일 밤 집무실을 찾아 증거를 압수했다.']
lm.predict(test)
