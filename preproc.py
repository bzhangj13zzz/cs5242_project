from urllib.parse import unquote
from deep_translator.google_trans import GoogleTranslator
import xmltodict
import pandas as pd
from deep_translator import MyMemoryTranslator

label_mapping = {1:"可回收", 2:"有害垃圾", 4:"湿垃圾", 8:"干垃圾", 16:"大件垃圾"}

def translate(src_txt):
    # print(src_txt)
    translator = GoogleTranslator(source='chinese', target='english')
    translated = translator.translate_batch(src_txt)
    return translated

def parse_csv(csv_path):
    df = pd.read_csv(csv_path, names=["text", "label"])
    # file = csv.DictReader(open(csv_path))
    # print(set(df[1].tolist()))
    grouped_df = df.groupby("label")
    # grouped_lists = grouped_df["text"].apply(list)

    for idx, group in grouped_df:
        l = group['text'].tolist()
        translate(l)

def parse_xml(xml_path):
    d = xmltodict.parse(open(xml_path, 'r').read())
    urls = d['urlset']['url']
    locs = [url['loc'] for url in urls if 'sk' in url['loc']]
    for loc in locs:
        print(loc)
        print(loc.split('/'))
        print(unquote(loc.split('/')[-1]))
        exit()

def translate_garbage_names(csv_path):
    df = pd.read_csv(csv_path, names=["cn", "label"])
    chinese_names = df['cn'].tolist()
    translated = translate(chinese_names)
    df['en'] = translated
    df.to_csv('translated.csv')
    print(df)

if __name__ == "__main__":
    # parse_csv("/Users/zhangbowen/Documents/garbage-classification-data/garbage.csv")
    translate_garbage_names("/Users/zhangbowen/Documents/garbage-classification-data/garbage.csv")
    # parse_xml("/Users/zhangbowen/Downloads/sitemaps/sitemap.xml")