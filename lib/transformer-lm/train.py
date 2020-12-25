from lib.util import Config
from lib.next_word_prediction import NextWordPrediction

# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/lm.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

# load w2v model and train
lm = NextWordPrediction(dconf, mconf)
lm.train()

lm.save('trained.pth')
mconf.save(mconf_path)

test = ['또 하나 필요한 것은 훌륭한', '또 하나 필요한 것은 훌륭한 영어 ', '경찰은 월요일 밤 집무실을 찾아 증거를 ']
print(lm.test(test))