from PIL import Image,ImageDraw
from decision_video import font
from kitchen import ROOT,NO_ITEM,ONION,DISH,PLATED_SOUP

ASSETS = ROOT / 'pufferlib-5/resources/overcooked'


class OvercookedView:
    def __init__(self):
        self.textures = {}

    def texture(self, name):
        if name not in self.textures:
            self.textures[name] = Image.open(ASSETS / name).convert('RGBA').resize((100,100),Image.Resampling.NEAREST)
        return self.textures[name]

    def render(self, k):
        im = Image.new('RGB',(500,560),'#141d30')
        names={0:'floor',1:'counter',2:'pot',4:'onions',5:'serve',6:'counter',7:'dishes'}
        tags={2:'POT',4:'ONIONS',5:'SERVE',7:'PLATES'}
        for y in range(k.q('height')):
            for x in range(k.q('width')):
                tile=k.q('tile',x,y)
                tex=self.texture('terrain/'+names[tile]+'.png')
                im.paste(tex,(x*100,y*100),tex)
                if tile == 2:
                    pot_index=k.q('pot_at',x,y)
                    count=k.q('pot_count',pot_index)
                    state=k.q('pot_state',pot_index)
                    if count:
                        name=('objects/soup-onion-cooked.png' if state==2 else
                              f'objects/soup-onion-{count}-cooking.png')
                        soup=self.texture(name)
                        im.paste(soup,(x*100,y*100),soup)
                item=k.q('item_type',x,y)
                item_sprite={ONION:'onion',DISH:'dish',PLATED_SOUP:'soup-onion-dish'}.get(item)
                if item_sprite:
                    item_tex=self.texture('objects/'+item_sprite+'.png')
                    im.paste(item_tex,(x*100,y*100),item_tex)
                if tile in tags:
                    d=ImageDraw.Draw(im)
                    d.rectangle((x*100,y*100+80,x*100+99,y*100+99),fill='#18202e')
                    d.text((x*100+5,y*100+81),tags[tile],font=font(12),fill='white')
        for i in reversed(range(k.num_agents)):
            direction=['NORTH','SOUTH','WEST','EAST'][k.q('facing',i)]
            suffix={NO_ITEM:'',ONION:'-onion',DISH:'-dish',PLATED_SOUP:'-soup-onion'}[k.q('held',i)]
            tex=self.texture('chefs/'+direction+suffix+'.png')
            x,y=k.q('x',i)*100,k.q('y',i)*100
            im.paste(tex,(x,y),tex)
            d=ImageDraw.Draw(im)
            d.ellipse((x+3,y+3,x+26,y+26),fill='#0b1020',outline='#65e2b2' if i==0 else '#a9bad3',width=2)
            d.text((x+10,y+5),str(i),font=font(13),fill='white')
        p=k.state()['pots'][0]
        d=ImageDraw.Draw(im)
        progress=f"   {p['cook_progress']} of 20 ticks" if p['state']=='cooking' else ''
        d.text((8,511),f"Pot   {p['onions']} of 3 onions   {p['state'].title()}{progress}",font=font(18),fill='#ecf2fc')
        actor_label = ('Chef 0 and Chef 1 act independently' if k.num_agents > 1
                       else 'Chef 0 acts autonomously')
        d.text((8,537),actor_label,font=font(15),fill='#a9bad3')
        return im
