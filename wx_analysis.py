#-*- coding: utf-8 -*-

import re
from wxpy import *
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator
import gmplot
from matplotlib.patches import Polygon
from matplotlib.colors import rgb2hex

# 初始化机器人，扫码登录
bot = Bot()

# 获取所有好友
my_friends = bot.friends()
print(type(my_friends))

# count the numbers of sexes and convert it into DataFrame
def find_out_sex(my_friends):
    sex_dict = {'male':0,'female':0,'other':0}
    for friend in my_friends:
        if friend.sex == 1:
            sex_dict['male'] += 1
        elif friend.sex == 2:
            sex_dict['female'] += 1
        else:
            sex_dict['other'] += 1
    
    df_sex = pd.DataFrame(list(sex_dict.items()),columns=['sex','count'])
    total = df_sex['count'].sum()
    df_sex['pct'] = df_sex['count']/total
    
    return df_sex

df_sex = find_out_sex(my_friends)

sns.set_palette('Set2')
# draw pie charts
def draw_pie(data):
	fig = plt.figure(figsize=(12,6))

	labels = data.sex  #use the sex as lables
	plt.axes(aspect=1)  # set this to make a round figure

	plt.title("性别情况",fontdict={'fontsize':15},loc='center')  

	plt.pie(data=data, x=data['pct'], labels=labels, autopct='%3.1f %%', radius=0.8,
	        labeldistance=1.1, startangle = 90, pctdistance = 0.7)
	plt.show()

draw_pie(df_sex)

# pull location data into DataFrame
def obtain_city_data(my_friends):
    # create cities list to store unique city names
    cities = []
    for friend in my_friends:
        if friend.city not in cities:
            cities.extend([friend.city])
    
    # create a city dictionary for number count
    cities_dict= {}
    for i in cities:
        cities_dict[i] = 0
    
    # pull the numbers out into the city dictionary
    for friend in my_friends:
        cities_dict[friend.city] += 1
    
    # convert the dictionary data into DataFrame
    df_city = pd.DataFrame(list(cities_dict.items()),columns=['city','count'])
    df_city['city'] = df_city['city'].apply(lambda x: 'Others' if x=='' else x)
    
    # groupby column city in case there's duplicate city items.
    df_city_unique = df_city.groupby('city').sum()
    #print(df_city_unique['count'].sum())   #check the total number
    df_city_unique = df_city_unique.reset_index()  # make sure the data is still in DataFrame
    
    return df_city_unique

df_city_unique = obtain_city_data(my_friends) 

# read lat/log of cities
city_lt = pd.read_csv('/Users/mabelfan/Documents/Ipython/城市经纬度.csv')
city_lt_copy = city_lt.copy()

# merge city_data with lat/log info
df_city_lt = df_city_unique.merge(city_lt_copy, how='left',\
                                  left_on = 'city', right_on ='city')

# delete the rows that still don't have lat and log information
df_city_lt_nonna = df_city_lt.dropna()

print("除包含经纬度为空时的数量:", df_city_lt['count'].sum(),'\n',\
	"去除空值之后的数量:", df_city_lt_nonna['count'].sum())

# draw friends distribution by city
def draw_distribution_city(data_a):
	fig = plt.figure(figsize=(12,8))

	lat = np.array(data_a['纬度'])                        # 获取维度之维度值
	lon = np.array(data_a['经度'])                        # 获取经度值
	pop = np.array(data_a['count'],dtype=float)    # 获取人口数，转化为numpy浮点型

	size=(pop/np.max(pop))*100    # 绘制散点图时图形的大小，如果之前pop不转换为浮点型会没有大小不一的效果

	map = Basemap(llcrnrlon=90, llcrnrlat=20, urcrnrlon=140,urcrnrlat=45,\
	              projection='stere', lat_0=35, lon_0=100, resolution='l')

	map.drawcoastlines()   
	map.drawcountries()    

	map.readshapefile('/Users/mabelfan/Documents/Ipython/wechatAnalysis/gadm36_CHN_shp/gadm36_CHN_1', 'states', drawbounds=True)
	map.readshapefile('/Users/mabelfan/Documents/Ipython/wechatAnalysis/gadm36_CHN_shp/gadm36_CHN_2', 'cities', drawbounds=False)
	map.readshapefile('/Users/mabelfan/Documents/Ipython/wechatAnalysis/gadm36_TWN_shp/gadm36_TWN_2', 'taiwan2', drawbounds=False)
	#map.drawmapboundary()

	parallels = np.arange(0.,90,10.) 
	map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10) # 绘制纬线

	meridians = np.arange(80.,140.,10.)
	map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10) # 绘制经线

	# compute map projection coordinates for lat/lon grid.
	x,y = map(lon,lat)

	# map.scatter(x,y,edgecolors='r',facecolors='r',marker='*',s=320)

	map.scatter(x,y,edgecolors='b',facecolors='b',marker='o',s=size)

	plt.title('全国好友按城市分布',fontdict={'fontsize':14},loc='center')

	plt.show()

draw_distribution_city(df_city_lt_nonna)

def make_pro_dict(my_friends):
	provinces = []
	for friend in my_friends:
		if friend.province not in provinces:
			provinces.extend([friend.province])

	provinces = [x for x in provinces if len(x)>0]    # delete null values

	pro_dict = {}
	for i in provinces:
		pro_dict[i] = 0
	return pro_dict

pro_dict = make_pro_dict(my_friends)

def make_pro_df(my_friends, dicts):
	for friend in my_friends:
		if friend.province in dicts.keys():    # in case there's null value cause problem
			dicts[friend.province] += 1

	df_pro = pd.DataFrame(list(dicts.items()), columns=['province','count'])
	df_pro['省名'] = df_pro['province'].apply([lambda x: x.replace(" ","")])
	df_pro.set_index('省名', inplace=True)
	return df_pro

df_pop = make_pro_df(my_friends, pro_dict)

def get_pro(info):
    pro = info.split('|')
    if len(pro) > 1:
        s = pro[1]
    else:
        s = pro[0]
    s = s[:2]
    if s == '黑龍':
        s = '黑龙江'
    if s == '内蒙':
        s = '内蒙古'
    return s

# attention! make sure to locate the two nessary py files in the same folder
from langconv import *

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    return Converter('zh-hans').convert(sentence)

def draw_distribution_pro(df_p):
	fig = plt.figure(figsize=(16,9))

# only show China map
	m = Basemap(llcrnrlon=77, llcrnrlat=14, urcrnrlon=140, urcrnrlat=51, \
		projection='lcc', lat_1=33, lat_2=45, lon_0=100)

# readshapefile, make sure which file you want to read in the gadm36_CHN_shp folder
	m.readshapefile('/Users/mabelfan/Documents/Ipython/wechatAnalysis/gadm36_CHN_shp/gadm36_CHN_1', 'states', drawbounds=True)

	m.readshapefile('/Users/mabelfan/Documents/Ipython/wechatAnalysis/gadm36_TWN_shp/gadm36_TWN_0', 'taiwan', drawbounds=True)

	cmap = plt.cm.YlOrRd
	vmax = 1500
	vmin = -10

	ax = plt.gca()
	for info, shp in zip(m.states_info, m.states):
	    state = info['NL_NAME_1']
	    proid = get_pro(state)
	    if proid not in df_p['province']:
	        pop = 0
	    else:
	        pop = df_p['count'][proid]*10
	    color = rgb2hex(cmap(np.sqrt((pop - vmin) / (vmax - vmin)))[:3])
	    poly = Polygon(shp,facecolor=color,edgecolor=color)
	    ax.add_patch(poly)

	for nshape, shp in enumerate(m.taiwan):
	    pop = df_p['count']['台湾']*10
	    color = rgb2hex(cmap(np.sqrt((pop - vmin) / (vmax - vmin)))[:3])
	    poly = Polygon(shp,facecolor=color,edgecolor=color)
	    ax.add_patch(poly)

	plt.title('全国好友近省份分布',fontdict={'fontsize':14},loc='center')
    
	plt.show()

draw_distribution_pro(df_pop)

def obtain_signatures(my_friends):
	#delete "span"，"class"，"emoji"，"emoji1f3c3" etc.
	signatures = []
	for i in my_friends[1:]:
		signature = i.signature.strip().replace("span", "").replace("class", "").replace("emoji", "")

		reg1 = re.compile(r"1f\d.+")
		reg2 = re.compile(r'\s*') #匹配占位符
		signature = reg2.sub("", reg1.sub("", signature))
		if len(signature) > 0:
			signatures.append(signature)

	# join all the signature strs
	full_sign = "".join(signatures)
	return full_sign

full_sign = obtain_signatures(my_friends)

# use jieba to cut sentences
def cut_to_words(text):
	stopwords = {}.fromkeys([line.rstrip() for line in open('/Users/mabelfan/Documents/Ipython/stopwords.txt','r')])
	wordlist_jieba = jieba.cut(text, cut_all=False)

	#wl_space_split = " ".join(wordlist_jieba)

	final = ''
	for word in wordlist_jieba:
		if word not in stopwords:
			final += word
			final += " "
	return final

final_sign = cut_to_words(full_sign)

def draw_wordcloud(text):
	plt.rcParams['font.sans-serif']=['Microsoft YaHei'] 

	# Generate a word cloud image
	font = r'/Library/Fonts/msyh.ttf'
	wordcloud = WordCloud(background_color="white",font_path=font, 
	                        width=1800, height=1200,max_words=2000, max_font_size=800,).generate(text)
	fig = plt.figure(figsize=(16,9))

	# Display the generated image:
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

draw_wordcloud(final_sign)

#end

























