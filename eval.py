# %%
from dearth_model import DearthForCausalLM, DearthConfig
import torch
import transformers
import yaml
import os

# %%
hf_cache_dir = "../hf_cache"
#model_name = "HuggingFaceH4/zephyr-7b-alpha"
model_name = "roneneldan/TinyStories-28M"
tk = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)

# %%
# text = """Microsoft finally releases Internet Explorer 10 for Windows 7, download it now Internet Explorer 10 (IE10) has been available for Windows 8 since the OS launch, but Windows 7 users were stuck with IE9 until now. This morning Microsoft released Internet Explorer 10 for Windows 7, which brings forth greater support for HTML5, improved speed and better privacy protection for users. IE10 is said to be about 20% faster than its predecessor IE9, and utilizes DirectX 11 for the browser's graphics hardware acceleration speed-ups. HTML5 support is improved by more than 60% bringing forth a wealth of new feature rich web elements that developers will be sure to take advantage of. Another feature worth noting is that Do Not Track is enabled by default. This feature blocks certain sites from tracking your browsing habits, such as Google who uses your browsing history to serve up targeted ads. Users must be running Windows 7 with Service Pack 1 installed in order to install Internet Explorer 10. You can download IE10 from the source link below. How do you feel about the new IE10? Let us know in the comments. Related Tags Further Reading: Read and find more Internet Browsers news at our Internet Browsers news index page. Do you get our news RSS feed? Get It! Got a news tip? Tell Us!"""
# text += """Honda Replaces IMA Hybrid Technology with New Sport Hybrid Systems More than a decade later, Honda continues to use IMA technology in its hybrid vehicles. Such systems, which cannot propel the vehicle on electricity alone, are known as mild hybrids. A full-hybrid vehicle has a gasoline electric hybrid powertrain that can operate solely on the electricity stored in the battery pack for short distances and at lower vehicle speeds. However, this is about to change with the 2014 Honda Accord Plug-in Hybrid that arrives in January 2013. As part of the company's Earth Dreams engine initiative, Honda is introducing a new lineup of Sport Hybrid powertrains for its hybrid-powered vehicles, and confirms that the two more popular systems will offer full-hybrid, EV driving capability. Sport Hybrid Intelligent Dual-Clutch Drive This new full-hybrid powertrain, which was announced on November 12, utilizes a new Earth Dreams 1.5-liter, Atkinson-cycle 4-cylinder engine, 7-speed dual-clutch transmission (DCT), single electric assist motor, and a Lithium-ion battery. Honda says that with this new hybrid powertrain, "the fun of driving is realized with acceleration g-force more powerful than that of existing models as well as a rhythmic and linear acceleration feeling." The automaker also says the EV driving mode is operational at vehicle start-up and when cruising at low-to-moderate speeds. Sport Hybrid Intelligent Multi-Mode Drive This is the new full-hybrid powertrain installed in the 2014 Honda Accord Plug-in Hybrid. The model is equipped with a new Earth Dreams 2.0-liter, Atkinson-cycle 4-cylinder engine, continuously variable transmission (CVT), two electric assist motors, and a Lithium-ion battery. The system automatically engages one of three driving modes, depending on how the vehicle is driven and the battery's state of charge: EV Drive, Hybrid Drive, and Engine Drive. In the Accord, this Sport Hybrid system is a plug-in hybrid, with a battery that offers pure electric driving for shorter trips. When the battery reaches a minimum state of charge, the system engages the gasoline engine and the vehicle operates as a conventional gasoline electric hybrid. Sport Hybrid SH-AWD Introduced in the Acura NSX Concept and destined for the redesigned 2014 Acura RLX luxury sedan, the new Sport Hybrid SH-AWD system pairs a direct-injection V-6 engine with a 7-speed DCT, three electric motors, and a Lithium-ion battery. One electric assist motor is contained within the transmission itself and directs power to the front and rear wheels. Two additional electric assist motors are located at the rear of the vehicle, controlling torque distribution to the rear wheels for a performance-oriented driving experience, according to Honda. Honda Model Ratings Honda Accord Ratings Acura Ratings"""
# text += """SE3 Condenser Microphone from SE Electronics Sonic Distribution is now handling the SE Electronics line of imported studio condensers. The SE3 caught my eye at the Summer NAMM Show in Nashville and is their flagship "pencil" microphone with a fixed cardioid pattern and 48V phantom powering. This mic uses Class A FET amplifier electronics and has both low cut filter and -10dB pad switches. I had the opportunity to try this mic out on several sources while recording a band and was impressed by its natural sound and all around usefulness. I used it for acoustic guitar overdubs where the low cut filter helped to tame a jumbo bodied guitar's boomy sound. The gentle presence lift added a sparkle without using EQ. I also tried it on drums and cymbals and it (using the pad) didn't fold up (overload) at all. I even tried it on vocals with good results although it does 'pop' easily and required a couple of pop screens. Housed in an elegantly finished new body design, it comes with a sturdy shock mount and packaged in a deluxe wooden travel case. Significant specifications are: frequency response rated at 20Hz-20khz; sensitivity is 10mV/Pa +/- 2dB; noise level is 17dB (A weighted); and Max SPL for 0.5% THD @ 1kHz is 135dB. I certainly found a 'Swiss army knife' of a condenser with the SE3 and I completely recommend it for any studio task especially acoustic instruments such as guitar, violin, cello or string bass. The SE3 sells for $349 MSRP and for much more contact Sonic Distribution at 617-623-5581 or go to: Web Page design is copyright © 2004 by Barry Rudolph"""
# text += """First I have my younger students trace a simple fish template, however my older students usually feel more confident so I have them hand draw out a fish shape. Then they draw an eye and lips if they would like to. Also, students are to draw a "fin" in the corner using any kind of line they would like...straight, curving, wiggly, etc. Then the fun part... adding patterns. Students draw patterns using markers and/or crayons to fill their entire fish. The next class we first finish our patterns and make sure we don't forget to color our fin in the corner too! Students then cut out the fish and the fin. I help the younger ones with this next step. Students cut a curving line into the fish body to create a "gill". We need it to curve away from the mouth. Then take the front half of the cut body and slide it over the back to overlap. This "pops" out the body. I will staple for the younger kids, however older ones can glue with a dot. Then the fin receives a dot of glue and is slid into this Popped gill to dry. Lastly students glue the fish onto a piece of 9 x 12" construction paper in whatever color they would like. We just put a dot of glue on the dorsal fin, mouth, and tail. No need for it anywhere else. They can use crayons and markers to create an environment around their fish. As you can see my students can come up with some really awesome patterns. These examples are created by 3rd grade students. I love this project. Simple 3-d and my students are always so proud of their works. Because of storage space, students nearly always take this project home on the second week after completion. This was a fun way to begin our break. This project took 2 forty minute classes to complete. Our materials were the following: 1. 9 x 12" white drawing paper 2. 9 x 12" color construction paper 3. Markers 4. Crayons 5. Scissors 6. Glue 7. Pencils and erasers 8. Templates (if you so choose to)"""
# print(len(tk.encode(text)))
# print(len(text))

# %%
def get_latest_ckpt(model_dir):
    ckpt_dir = os.path.join(model_dir)
    if not os.path.exists(ckpt_dir):
        return None
    if not os.path.isdir(ckpt_dir):
        return None
    ckpt_list = os.listdir(ckpt_dir)
    if len(ckpt_list) == 0:
        return None
    import re
    num_sort_func = lambda s: sum(((s,int(n))for s,n in re.findall('(\D+)(\d+)','a%s0'%s)),()) # from https://cloud.tencent.com/developer/article/1856550
    ckpt_list.sort(key=num_sort_func)
    return os.path.join(ckpt_dir, ckpt_list[-1])

# %%
model_dir = "./ckpt/ts100-re2"
#model_dir = "./ckpt/ts42-re1"
model_path = get_latest_ckpt(model_dir)
#model_path = "./ckpt/ts70/ts70-1000.pt"
print(model_path)
yml_path = "./config/ts100-re2.yml"
with open(yml_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)['model']
if "vocab_size" not in config:
    config['vocab_size'] = tk.vocab_size
config["attn_window_size"] = 500
print(config)
config = DearthConfig(**config)
model = DearthForCausalLM(config)
states = torch.load(model_path, map_location="cpu")
print(states.keys())
model_states = states["model"]
unwanted_prefix_dueto_compile = '_orig_mod.'
unwanted_prefix_dueto_ddp = 'module.'
unwanted_prefix_dueto_ddp_compiled = 'module._orig_mod.'

for k,v in list(model_states.items()):
    if k.startswith(unwanted_prefix_dueto_ddp_compiled):
        new_key = k[len(unwanted_prefix_dueto_ddp_compiled):]
        model_states[k[len(unwanted_prefix_dueto_ddp_compiled):]] = model_states.pop(k)
    elif k.startswith(unwanted_prefix_dueto_ddp):
        new_key = k[len(unwanted_prefix_dueto_ddp):]
        model_states[k[len(unwanted_prefix_dueto_ddp):]] = model_states.pop(k)
    elif k.startswith(unwanted_prefix_dueto_compile):
        new_key = k[len(unwanted_prefix_dueto_compile):]
        model_states[k[len(unwanted_prefix_dueto_compile):]] = model_states.pop(k)

model.load_state_dict(states["model"])

# %%
# print all params
name_set = set()
for n, p in model.named_parameters():
    percise_linear_name = n.split(".")[-2]
    name_set.add(percise_linear_name)

print(name_set)


# %%
#input_text = "The quick brown fox jumps over the lazy dog, "
input_text = "Once upon a time, there was a little girl"
input_ids = tk.encode(input_text)
input_ids = torch.tensor(input_ids).unsqueeze(0)

# %%
model.eval()

# count model parameters, excluding embeddings
num_params = 0
for n, p in model.named_parameters():
    if not "emb" in n:
        num_params += p.numel()
print(f"num params: {num_params}")

# %%
# generate more tokens
output_ids = input_ids.squeeze(0).tolist()
for i in range(32):
    input = torch.tensor(output_ids, dtype=torch.long).view(1, -1)
    with torch.no_grad():
        output = model(input)[0]
        tmp_output_ids = output.argmax(dim=-1)
    tmp_output_ids = tmp_output_ids.squeeze(0).tolist()
    if i == 0:
        output_ids = tmp_output_ids
    else:
        output_ids.append(tmp_output_ids[-1])

print(output_ids)
print(tk.decode(output_ids))


