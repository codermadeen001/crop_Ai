from bing_image_downloader import downloader

non_maize_classes = [
    # non-leaf everyday objects
    "car", "computer", "cow", "book", "keyboard", "sky", "chair", "table", "cup", "jet",
    "person", "mountain", "road", "tree", "bike", "building", "food", "flower", "cat", "dog",
    "paper", "cartoon", "watch", "teddybear", "phone", "television", "shoe", "pen", "bus",
    "bottle", "plant pot", "mirror", "sofa", "bed", "lamp", "curtain", "fridge", "street",
    "hands", "face", "clock", "powerline", "bookshelf", "hand", "jacket", "cloth", "arm",

    # other crop leaves
    "bean leaf", "cassava leaf", "tomato leaf", "potato leaf", "sorghum leaf",
    "banana leaf", "rice leaf", "cabbage leaf", "spinach leaf", "kale leaf",
    "lettuce leaf", "cotton leaf", "sunflower leaf", "avocado leaf", "millet leaf",
    "groundnut leaf", "napier grass leaf", "sugarcane leaf", "pawpaw leaf",
    "soybean leaf"
]

for category in non_maize_classes:
    downloader.download(category, limit=500, output_dir='non_maize_dataset', adult_filter_off=True, force_replace=False, timeout=60)
