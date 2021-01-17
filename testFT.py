import fasttext
from skimage import io
import jieba

model = fasttext.load_model("data_dim200_lr1_iter1000_ngram3.model")
meme_map = {0: 'http://mmbiz.qpic.cn/mmemoticon/ajNVdqHZLLCdwOLyl7c6JM4VOicLMB1H8ldGoG6pf6YSWq0ic3dP0hsI1yibXwJoHyF/0', 1: 'http://mmbiz.qpic.cn/mmemoticon/ajNVdqHZLLALxSLBT9ia2YKwibicvnJT1lcGSPseOJib9Iibic4PK3OgQhNiaPPDznicicTV0/0', 2: 'http://mmbiz.qpic.cn/mmemoticon/ajNVdqHZLLAqn9uGg0riawTIoqtibyenzzoeRRXpQT1ME9Og5jIEicEI39WMYbX1LLM/0', 3: 'http://mmbiz.qpic.cn/mmemoticon/Q3auHgzwzM5llZ8F17rYSxaT0rHd1ibtq83VqoK8HNiadrBn4c79H7IDZqxCzKTD98/0', 4: 'http://mmbiz.qpic.cn/mmemoticon/Q3auHgzwzM7lf1vRwFDtwHoCmelDVIGiaLB6x8wMtR0yiaNQ7lK4LUXfsf2hIpex9D/0', 5: 'http://mmbiz.qpic.cn/mmemoticon/ajNVdqHZLLBVWrxFJdiaibzrsehMjZZU1xtm9xTUacErlzRLK7jbibuo0pSRES1IoCo/0', 6: 'http://mmbiz.qpic.cn/mmemoticon/ajNVdqHZLLCSGzU8cqeRmFMDDiaw2KKbpJfic0AqHnkugYtKRbdUznnF9uUjcW7sM9/0', 7: 'http://mmbiz.qpic.cn/mmemoticon/ajNVdqHZLLCdwOLyl7c6JNk0zs9dNFASHTt5adwwV6ZcaO3ImnSwjnDj5LgBvjxy/0', 8: 'http://mmbiz.qpic.cn/mmemoticon/ajNVdqHZLLAhXBBjIqSl2TVadDMGVSJN7H9Bty3NjyiaE7ibCuibHcLz07kMRWT0g7v/0', 9: 'http://mmbiz.qpic.cn/mmemoticon/ajNVdqHZLLC3ib2qk5SSCGLN1GGzwwphbh5ec1NXqLPsB0lZ6woHJgxvgOsLKJNvg/0'}
demo = ["可以麻烦帮我烧一下水嘛", "nlp让我焦头烂额", "今天天气好好", "写一首诗给我看看", "能不能帮她写个代码"]
demo_seg = []
for sent in demo:
    seg_list = jieba.cut(sent)
    demo_seg.append(' '.join(list(seg_list)))
for i, sent in enumerate(demo_seg):
    label = model.predict(sent)[0][0]
    demo_url = meme_map[int(label[9])]
    print(demo[i])
    print(int(label[9]))
    #image = io.imread(demo_url)
    #io.imshow(image)
    #io.show()
    
