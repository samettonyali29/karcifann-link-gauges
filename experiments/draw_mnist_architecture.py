"""Draw the MNIST network as vector SVG/PDF without loading data or training.

Run: python experiments/draw_mnist_architecture.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]


def main():
    OUT = ROOT / 'figures'
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'karcifann-figure-1',
    })

    fig = plt.figure(figsize=(11.8, 4.45), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0,1180), ylim=(520,75), aspect='equal')
    ax.axis('off')
    ink, muted, rule = '#172B3A', '#536778', '#DCE4EA'
    blue, teal, purple = '#245CAA', '#007F82', '#7752A2'

    def text(x,y,s,size=12,color=ink,weight='normal',ha='left',**kwargs):
        return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,
                       va='center',zorder=6,**kwargs)

    def box(x,y,w,h,fill,edge='none',radius=12,lw=.8,z=1):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                     facecolor=fill,edgecolor=edge,linewidth=lw,zorder=z))

    def arrow(x1,y1,x2,y2,color=muted,lw=1.1):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
                      mutation_scale=11,linewidth=lw,color=color,zorder=4))

    layers = [
        (175,blue,'#F0F5FC','INPUT','784 features',r'$\mathbf{x}\in\mathbb{R}^{784}$',[1,2,3,783,784]),
        (590,teal,'#EDF8F6','HIDDEN','50 sigmoid units',r'$\mathbf{h}\in(0,1)^{50}$',[1,2,3,49,50]),
        (1005,purple,'#F5F0F9','OUTPUT','10 sigmoid units',r'$\hat{\mathbf{y}}\in(0,1)^{10}$',[1,2,3,9,10]),
    ]
    ys=[207,249,291,364,406]

    for x,color,pale,title,count,symbol,indices in layers:
        text(x,93,title,10.8,color,'bold',ha='center')
        text(x,121,count,17,ink,'bold',ha='center')
        box(x-96,154,192,284,pale,radius=15)

    # Every displayed unit connects to every displayed unit in the next layer.
    for left,right,col in [(175,590,blue),(590,1005,teal)]:
        for a in ys:
            for b in ys:
                ax.plot([left+20,right-20],[a,b],color=col,alpha=.16,lw=.72,zorder=2)
        text((left+right)/2,160,'Fully connected',10.5,muted,ha='center')
        arrow(left+132,179,right-132,179,color=muted,lw=.85)

    for x,color,pale,title,count,symbol,indices in layers:
        for y,idx in zip(ys,indices):
            ax.add_patch(Circle((x,y),19.5,facecolor='white',edgecolor=color,
                                linewidth=1.5,zorder=5))
            text(x,y,str(idx),10.8 if idx>=100 else 12,color,'bold',ha='center')
        for y in (316,326,336):
            ax.add_patch(Circle((x,y),1.65,facecolor=color,edgecolor='none',zorder=5))
        text(x,460,symbol,15,color,ha='center')

    text(175,493,r'$28\times28$ pixels, flattened',11,muted,ha='center')
    text(590,493,r'$\mathbf{h}=\sigma(W_1\mathbf{x}+\mathbf{b}_1)$',14,teal,ha='center')
    text(1005,493,r'$\hat{\mathbf{y}}=\sigma(W_2\mathbf{h}+\mathbf{b}_2)$',14,purple,ha='center')


    for extension in ('svg','pdf'):
        file = OUT / f'architecture_mnist.{extension}'
        metadata = ({'Title':'Figure 1. Shared classification architecture',
                     'Description':'MNIST 784-50-10 fully connected network; sigmoid hidden and output units; minibatch MSE.',
                     'Date':None} if extension=='svg' else
                    {'Title':'Figure 1. Shared classification architecture',
                     'Author':'Samet Tonyali', 'CreationDate':None,'ModDate':None})
        fig.savefig(file,format=extension,metadata=metadata,facecolor='white')
    plt.close(fig)
    print('Created figures/architecture_mnist.svg and .pdf.')


if __name__ == "__main__":
    main()
