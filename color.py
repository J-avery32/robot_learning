from colored import Fore, Back, Style
import colored

from pathlib import Path

txt = Path("map_path.txt").read_text()

print(txt.replace("[", " ").replace("]"," ").replace("1.", f'{Back.red}1{Style.reset}').replace("0.", f'{Back.blue}0{Style.reset}').replace("8.", f'{Back.green}8{Style.reset}'))