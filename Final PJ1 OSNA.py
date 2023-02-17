import tweepy as tw
from tweepy import API
from collections import defaultdict
from dotenv import load_dotenv
from pynput import keyboard
import json
import os
import io
load_dotenv("config.env")

# Twitter authentication
auth = tw.OAuthHandler(
    os.environ["api_key"],
    os.environ["api_key_secret"],
)
auth.set_access_token(
    os.environ["access_token"],
    os.environ["access_token_secret"],
)

# Creating an API object, the api will wait if the rate limit exceeds
# NOTE:- API rate limit is 180 calls every 15 mins
# wait_on_rate_limit=True use this to apply sleep of 15 min 
api = tw.API(auth,wait_on_rate_limit=True)

def startupCheck():
    PATH = "./data.json"
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        # checks if file exists
        print("File exists and is readable")
    else:
        print("Either file is missing or is not readable, creating file...")
        with io.open("data.json", "w") as db_file:
            # profile_link will contain all users and thier profile picture
            # connections is adjascency list to form a network
            # screen_names will act as queue in order to expand the network
            db_file.write(
                json.dumps(
                    {
                        "profile_link": {},
                        "connections": {},
                        "starting_name": "ps_pujansheth",
                        "screen_names": ["ps_pujansheth"],
                    }
                )
            )


startupCheck()

# Loading the newly written or preloaded json file
with open('./data.json', 'r') as file:
    data = json.load(file)

# Get the first user deatils and store the profile image
fp_data=api.get_user(screen_name=data['starting_name'])
if data['starting_name'] not in data['profile_link']:
    data['profile_link'][data['starting_name']] = fp_data.profile_image_url_https
connections = defaultdict(list)

print(data['profile_link'])

flag = True

# Get at most 10 followers for a given node
# Network starts with me
def get_all_followers(screen_name):
    followers = []
    try:
        for follower in tw.Cursor(
            method=api.get_followers, screen_name=screen_name
        ).items(10):
            data["profile_link"][
                follower.screen_name
            ] = follower.profile_image_url_https
            followers.append(follower.screen_name)
            if follower.screen_name not in data["screen_names"]:
                data["screen_names"].append(follower.screen_name)
    except Exception:
        return _extracted_from_get_all_followers_14()
    return followers


# TODO Rename this here and in `get_all_followers`
def _extracted_from_get_all_followers_14():
    global flag
    print(Exception)
    print("Leaving the program due to exception")
    print("Saving the connections to the json file")
    data["connections"] = {**data["connections"], **connections}
    with open("./data.json", "w") as file:
        file.write(json.dumps(data))
    flag = False
    return 'stop'


# Run until no of nodes/profiles not exceed 500
def streaming():
    if len(data['connections']) >= 500:
        return "Gathered the maximum number of users/nodes allowed"
    for name in data["screen_names"]:
        if name not in data["connections"]:
            print(f"Adding a node with screen name {name}")
            response = get_all_followers(name)
            if response == 'stop':
                return 'Stopped the streaming due to an error'
            else: 
                connections[name] = response
    return f"Got a total of {len(data['profile_link'])} users"


def main():
    while flag:
        print(f"Starting at {data['starting_name']}")
        if len(data["profile_link"]) > 501:
            return "Gathered the maximum number of users/nodes allowed which is equal to 500"
        print(streaming())
    return "Stopped the streaming"


print(main())

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import urllib.request
from PIL import Image

# Generating nodes
G = nx.Graph()
for name, profile_pic in data['profile_link'].items():
    G.add_node(name, image=np.array(Image.open(urllib.request.urlopen(profile_pic))))
print(G.nodes())

# Adding an edge to signify connection between nodes aka users
for parent, followers in data['connections'].items():
    for follower in followers:
        G.add_edge(parent, follower)
print(G.edges())


pos = nx.circular_layout(G)

fig = plt.figure(figsize=(75, 75))
ax = plt.subplot(111)
ax.set_aspect("equal")
nx.draw_networkx_edges(G, pos, ax=ax)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

piesize = 0.01  # this is the image size
p2 = piesize / 2.0
for n in G:
    xx, yy = trans(pos[n])  # figure coordinates
    xa, ya = trans2((xx, yy))  # axes coordinates
    a = plt.axes([xa - p2, ya - p2, piesize, piesize])
    a.set_aspect("equal")
    a.imshow(G.nodes[n]["image"])
    a.axis("off")
ax.axis("off")
plt.show()

G.degree

dic_closeness = nx.closeness_centrality(G)
plt.bar(dic_closeness.values(), dic_closeness.keys(), 0.002)


degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)

fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# Create a gridspec for adding subplots of different sizes
axgrid = fig.add_gridspec(5, 4)

ax0 = fig.add_subplot(axgrid[0:3, :])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc, seed=10396953)
nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
ax0.set_title("Connected components of G")
ax0.set_axis_off()

ax1 = fig.add_subplot(axgrid[3:, :2])
ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Degree Rank Plot")
ax1.set_ylabel("Degree")
ax1.set_xlabel("Rank")

ax2 = fig.add_subplot(axgrid[3:, 2:])
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")

fig.tight_layout()
plt.show()

# largest connected component
components = nx.connected_components(G)
largest_component = max(components, key=len)
H = G.subgraph(largest_component)

# compute centrality
centrality = nx.betweenness_centrality(H, k=10, endpoints=True)

# compute community structure
lpc = nx.community.label_propagation_communities(H)
community_index = {n: i for i, com in enumerate(lpc) for n in com}

#### draw graph ####
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(H, k=0.15, seed=4572321)
node_color = [community_index[n] for n in H]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    H,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

# Title/legend
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Gene functional association network (C. elegans)", font)
# Change font color for legend
font["color"] = "r"

ax.text(
    0.80,
    0.10,
    "node color = community structure",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = betweenness centrality",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)

# Resize figure for label readability
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.show()

centrality
