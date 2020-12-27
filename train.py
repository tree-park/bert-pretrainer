
from transformer_lm.lib.util import Config
from lib.bert_pretrain import BERTEmbedding

# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/lm.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

# load w2v model and train
lm = BERTEmbedding(dconf, mconf)
lm.train()

lm.save('trained.pth')
mconf.save(mconf_path)

