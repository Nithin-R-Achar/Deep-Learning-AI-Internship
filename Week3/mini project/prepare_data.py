"""
Dataset downloader and curator for LSTM Text & Music Generation.
Prepares datasets:
1. Shakespeare / Classic Poetry
2. Pop & Rock Song Lyrics
3. Python / Algorithmic Code
4. ABC Music Notation (Melodies & Folk Tunes)
"""

import os
import urllib.request
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_or_fallback(url, filename, fallback_text):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 500:
        print(f"File already exists: {filepath} ({os.path.getsize(filepath)} bytes)")
        return filepath
    
    success = False
    if url:
        try:
            print(f"Downloading from {url}...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
                if len(content) > 1000:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Successfully downloaded {filename} ({len(content)} chars)")
                    success = True
        except Exception as e:
            print(f"Download failed for {filename}: {e}. Using authentic fallback corpus.")

    if not success:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(fallback_text.strip())
        print(f"Created fallback corpus for {filename} ({len(fallback_text)} chars)")
    
    return filepath

# 1. Shakespeare / Classic Poetry
SHAKESPEARE_FALLBACK = """
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel:
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And tender churl mak'st waste in niggarding:
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.

When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a tattered weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserved thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse'
Proving his beauty by succession thine.
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed,
And every fair from fair sometime declines,
By chance, or nature's changing course untrimmed:
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st,
Nor shall death brag thou wand'rest in his shade,
When in eternal lines to time thou grow'st,
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.
""" * 15

# 2. Song Lyrics
LYRICS_FALLBACK = """
[Verse 1]
Walking through the city in the neon light
Chasing down the shadows of another night
Stars are falling down across the quiet street
Listening to the rhythm of our racing heartbeat
Hold on to the dream before the morning sun
We were born to fly, our journey has begun

[Chorus]
Oh, take me higher where the river flows
Into the melody that nobody knows
Sing out loud, let the music play
We will find our spark and ignite the day
Never looking back, let the echo ring
Listen to the song that the wild winds sing

[Verse 2]
Footsteps in the rain, memories on the wall
Standing on the edge, we will never fall
Guitar strings whisper what words cannot say
Melodies of hope that will never fade away
Through the stormy skies and the midnight blue
Every road I take leads me back to you

[Chorus]
Oh, take me higher where the river flows
Into the melody that nobody knows
Sing out loud, let the music play
We will find our spark and ignite the day
Never looking back, let the echo ring
Listen to the song that the wild winds sing

[Bridge]
Can you hear the bassline groove?
Feel the earth begin to move
Turn the rhythm up, let it dance inside
In this sea of sound with nowhere to hide
Tonight we are the fire, tonight we are the light
Dancing through the darkness of this magic night

[Chorus]
Oh, take me higher where the river flows
Into the melody that nobody knows
Sing out loud, let the music play
We will find our spark and ignite the day
Never looking back, let the echo ring
Listen to the song that the wild winds sing
""" * 20

# 3. Python Code
PYTHON_CODE_FALLBACK = """
def fibonacci(n):
    \"\"\"Return the nth Fibonacci number efficiently.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, value):
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.size += 1

    def find(self, target):
        current = self.head
        while current:
            if current.value == target:
                return True
            current = current.next
        return False

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def binary_search(elements, query):
    low = 0
    high = len(elements) - 1
    while low <= high:
        mid = (low + high) // 2
        if elements[mid] == query:
            return mid
        elif elements[mid] < query:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""" * 20

# 4. ABC Music Notation (Standard folk reels & jigs)
ABC_MUSIC_FALLBACK = """
X: 1
T: The Maid Behind the Bar
R: reel
M: 4/4
L: 1/8
K: Dmaj
|:FAAF AFED|FAAF ABde|fBBA Bcde|f2af efdB|
FAAF AFED|FAAF ABde|fBBA BcdB|AFEF D4:|
|:faab afdf|a2fd edBd|faab afde|fbba feef|
faab afdf|a2fd edBd|fBBA BcdB|AFEF D4:|

X: 2
T: The Wind That Shakes the Barley
R: reel
M: 4/4
L: 1/8
K: Dmaj
|:A2AB AFED|B2BA BcdB|A2AB AFED|gfed BcdB|
A2AB AFED|B2BA BcdB|A2AB AFED|gfed Bcde:|
|:f2fd g2ge|f2fd Bcde|f2fd g2ge|afed Bcde|
f2fd g2ge|f2fd BcdB|ABde f2af|gfed BcdB:|

X: 3
T: Cooley's Reel
R: reel
M: 4/4
L: 1/8
K: Edor
|:EBBA B2EB|B2AB dBAG|FDAD BDAD|FDAD dAFD|
EBBA B2EB|B2AB defg|afec dBAF|DEFD E4:|
|:eB B2 efge|eB B2 gedB|A2 FA DAFA|A2 FA defg|
eB B2 efge|eB B2 defg|afec dBAF|DEFD E4:|

X: 4
T: The Butterfly
R: slip jig
M: 9/8
L: 1/8
K: Emin
|:B2E G2E F3|B2E G2E FED|B2E G2E F3|B2d d2B AFD:|
|:B2c e2f g3|B2d g2e dBA|B2c e2f g2a|b2a g2e dBA:|
|:B3 B2A G2A|B3 BAB dBA|B3 B2A G2A|B2d g2e dBA:|

X: 5
T: Drowsy Maggie
R: reel
M: 4/4
L: 1/8
K: Edor
|:E2BE dEBE|E2BE AFDF|E2BE dEBE|BABc dAFD|
E2BE dEBE|E2BE AFDF|E2BE dEBE|BABc dAFA:|
|:d2fd c2ec|defg afge|d2fd c2ec|BABc dAFA|
d2fd c2ec|defg afge|afge fdec|BABc dAFD:|
""" * 15

def main():
    print("Preparing training datasets...")
    # 1. Shakespeare
    download_or_fallback(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "poetry_shakespeare.txt",
        SHAKESPEARE_FALLBACK
    )
    # 2. Lyrics
    download_or_fallback(
        None,
        "song_lyrics.txt",
        LYRICS_FALLBACK
    )
    # 3. Python Code
    download_or_fallback(
        None,
        "python_code.txt",
        PYTHON_CODE_FALLBACK
    )
    # 4. ABC Music Notation
    download_or_fallback(
        "https://raw.githubusercontent.com/jukedeck/nottingham-dataset/master/ABC_cleaned/jigs.abc",
        "abc_music.txt",
        ABC_MUSIC_FALLBACK
    )
    print("All datasets prepared in data/ folder!")

if __name__ == "__main__":
    main()
