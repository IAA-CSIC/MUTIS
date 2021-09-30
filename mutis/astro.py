# Licensed under a 3-clause BSD style license - see LICENSE
"""Utils specific to the field of astrophysics"""

import logging

import numpy as np
import pandas as pd

import matplotlib as mplt
import matplotlib.pyplot as plt

import glob
import re
from datetime import datetime

from astropy.time import Time


__all__ = ["Astro"]

log = logging.getLogger(__name__)


def pol_angle_reshape(s):
    """
        Reshape a signal as a polarization angle: shifting by 180 degrees in a way it varies as smoothly as possible.
    """
    
    s = np.array(s)
    s = np.mod(s,180) # s % 180

    sn = np.empty(s.shape)
    for i in range(1,len(s)):
        #if t[i]-t[i-1] < 35:
        d = 181
        n = 0
        for m in range(0,100):
            m2 = (-1)**(m+1)*np.floor((m+1)/2)
            if np.abs(s[i]+180*m2-sn[i-1]) < d:
                d = np.abs(s[i]+180*m2-sn[i-1])
                n = m2
        sn[i] = s[i]+180*n
    return sn 






#########################################
################# Knots #################
#########################################



def  KnotsIdAuto(mod):
    """
       Identify the knots appearing in several epochs giving them names, based on their position.
       
        Parameters:
        -----------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot, with 
             at least columns 'label', 'date', 'X', 'Y', 'Flux (Jy)'.
       
        Returns: mod
        --------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot with their altered labels, with 
             at least columns 'label', 'date', 'X', 'Y', 'Flux (Jy)'.
         
    """
    
    mod_date_dict = dict(list((mod.groupby('date'))))
    
    mod_dates = list(Time(list(mod_date_dict.keys())).datetime)
    mod_dates_str = list(Time(list(mod_date_dict.keys())).strftime('%Y-%m-%d'))
    mod_data = list(mod_date_dict.values())
    
    Bx = [[]]*len(mod_dates)
    By = [[]]*len(mod_dates)
    B = [[]]*len(mod_dates)


    thresh = 0.03 #0.062

    news = 0

    for i, (date, data) in enumerate(zip(mod_dates, mod_data)):
        log.debug(f'Analysing epoch i {i:3d} ({mod_dates_str[i]})')

        if not len(data.index) > 0:
            log.error(' -> Error, no components found')
            break

        log.debug(f' it has {len(data.index)} components')

        Bx[i] = np.ravel(data['X'])
        By[i] = np.ravel(data['Y'])
        B[i] = np.full(len(data.index), fill_value=None)

        # if first epoch, just give them new names...:
        if i == 0:
            log.debug(' first epoch, giving names...')

            for n in range(0,len(mod_data[i].index)):
                if data['Flux (Jy)'].iloc[n] < 0.001:
                    log.debug('  skipping, too weak')
                    break
                if n == 0:
                    B[i][n] = 'A0'
                else:
                    news = news + 1
                    B[i][n] = f'B{news}'
            log.debug(f' -> FOUND: {B[i]}\n')
            continue

        # if not first epoch...:

        for n in range(0,len(mod_data[i].index)):
            if data['Flux (Jy)'].iloc[n] < 0.001:
                    log.debug('  skipping, too weak')
                    break
            if n == 0:
                B[i][n] = 'A0'
            else:
                log.debug(f' -> id component {n}...')

                close = None

                a = 0
                while ( i - a >= 0 and a < 4 and close is None):
                    a = a + 1

                    if not len(Bx[i-a])>0:
                        break

                    dist = ( (Bx[i-a]-Bx[i][n]) ** 2 + (By[i-a]-By[i][n]) ** 2 ) ** 0.5

                    for m in range(len(dist)):
                        if B[i-a][m] in B[i]:
                            dist[m] = np.inf

                    if np.amin(dist) < thresh*a**1.5:
                        close = np.argmin(dist)

                    if B[i-a][close] in B[i]:
                            close = None


                if close is None:
                    news = news + 1
                    B[i][n] = f'B{news}'
                    log.debug(f'   component {n} is new, naming it {B[i][n]}')
                else:
                    log.debug(f'   component {n} is close to {B[i-a][close]} of previous epoch ({a} epochs before)')
                    B[i][n] = B[i-a][close]


        log.debug(f' -> FOUND: {B[i]}\n')
        
                 
    for i, (date, data) in enumerate(zip(mod_dates, mod_data)):
        data['label'] = B[i]
         
    mod = pd.concat(mod_data, ignore_index=True)
            
    return mod



def KnotsId2dGUI(mod, use_arrows=False, arrow_pos=1.0):
    """
        Prompt a GUI to select identified knots and alter their label, reprenting their 2D
        spatial distribution in different times.
        
        It can be used inside jupyter notebooks, using '%matplotlib widget' first.
        
        Parameters:
        -----------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot, with at least columns 
             'label', 'date', 'X', 'Y', 'Flux (Jy)'.
        
        Returns:
        --------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot with their altered labels, with 
             at least columns 'label', 'date', 'X', 'Y', 'Flux (Jy)'.
    """
    
    mod = mod.copy()
    
    knots = dict(tuple(mod.groupby('label')))
    knots_names = list(knots.keys())
    knots_values = list(knots.values())
    knots_jyears = {k:Time(knots[k]['date'].to_numpy()).jyear for k in knots}
    knots_X = {k:knots[k]['X'].to_numpy() for k in knots}
    knots_Y = {k:knots[k]['Y'].to_numpy() for k in knots}


    from matplotlib.widgets import Slider, Button, TextBox, RectangleSelector

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

    lineas = list()
    flechas = list()
    textos = list()

    def draw_all(val=2008):
        nonlocal lineas, flechas, textos

        # instead of clearing the whole axis, remove artists
        for linea in lineas:
            if linea is not None:
                linea.remove()
        for texto in textos:
            if texto is not None:
                texto.remove()
        for flecha in flechas:
            if flecha is not None:
                flecha.remove()
        ax.set_prop_cycle(None)
        
        lineas = list()
        flechas = list()
        textos = list()

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        #ax.clear() # either clear the whole axis or remove every artist separetly
            
        for i, label in enumerate(knots_names):
            years = knots_jyears[label]
            idx = (val-1.5 < years) & (years < val)
            x = knots_X[label][idx]
            y = knots_Y[label][idx]

            lineas.append(ax.plot(x, y, '.-', linewidth=0.6, alpha=0.4, label=label)[0])

            if use_arrows:
                if len(x) > 1:
                    flechas.append(ax.quiver(x[:-1], 
                               y[:-1], 
                               arrow_pos*(x[1:] - x[:-1]), 
                               arrow_pos*(y[1:] - y[:-1]), 
                               scale_units='xy', angles='xy', scale=1, 
                               width=0.0015, headwidth=10, headlength=10, headaxislength=6,
                               alpha=0.5, color=lineas[i].get_color()))
                else:
                    flechas.append(None)

            if len(x) > 0:
                #textos.append(ax.annotate(label, (x[0], y[0]), (-28,-10), textcoords='offset points', color=lineas[i].get_color(), fontsize=14))
                textos.append(ax.text(x[-1]+0.015, y[-1]+0.015, label, {'color':lineas[i].get_color(), 'fontsize':14}))
            else:
                textos.append(None)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        
        fig.canvas.draw_idle() # if removed every artist separately instead of ax.clear()

    draw_all()

    def update(val):
        nonlocal lineas, flechas, textos

        for i, label in enumerate(knots_names):
            years = knots_jyears[label]
            idx = (val-1.5 < years) & (years < val)
            x = knots_X[label][idx]
            y = knots_Y[label][idx]

            lineas[i].set_xdata(x)
            lineas[i].set_ydata(y)

            if textos[i] is not None:
                if len(x) > 0:    
                    textos[i].set_position((x[-1]+0.015, y[-1]+0.015))
                    textos[i].set_text(label)
                else:
                    textos[i].remove()
                    textos[i] = None
                    #textos[i].set_position((10, 10))
            else:
                if len(x) > 0:    
                    textos[i] = ax.text(x[-1]+0.02, y[-1]+0.02, label, {'color':lineas[i].get_color(), 'fontsize':14})

            if use_arrows:
                if flechas[i] is not None:
                    flechas[i].remove()
                    flechas[i] = None

                flechas[i] = ax.quiver(x[:-1], 
                                   y[:-1], 
                                   arrow_pos*(x[1:] - x[:-1]), 
                                   arrow_pos*(y[1:] - y[:-1]), 
                                   scale_units='xy', angles='xy', scale=1, 
                                   width=0.0015, headwidth=10, headlength=10, headaxislength=6, 
                                   alpha=0.5, color=lineas[i].get_color())

        fig.canvas.draw_idle()



    selected_knot = None
    selected_ind = None
    selected_x = None
    selected_y = None


    def submit_textbox(text):
        nonlocal mod, knots, knots_names, knots_values, knots_jyears, knots_X, knots_Y

        log.debug('Submited with:')
        log.debug(f'   selected_knot {selected_knot}')
        log.debug(f'   selected_ind {selected_ind}')
        log.debug(f'   selected_x {selected_x}')
        log.debug(f'   selected_y {selected_y}')

        if selected_knot is not None:        
            mod.loc[selected_ind, 'label'] = text.upper()

            knots = dict(tuple(mod.groupby('label')))
            knots_names = list(knots.keys())
            knots_values = list(knots.values())
            knots_jyears = {k:Time(knots[k]['date'].to_numpy()).jyear for k in knots}
            knots_X = {k:knots[k]['X'].to_numpy() for k in knots}
            knots_Y = {k:knots[k]['Y'].to_numpy() for k in knots}

            print(f"Updated index {selected_ind} to {text.upper()}")
        else:
            pass

        draw_all(slider_date.val)

    def line_select_callback(eclick, erelease):
        nonlocal selected_knot,selected_x, selected_y, selected_ind

        # 1 eclick and erelease are the press and release events
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        log.debug("GUI: (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        log.debug("GUI:  The button you used were: %s %s" % (eclick.button, erelease.button))

        selected_knot = None
        selected_x = None
        selected_y = None
        selected_ind = None

        for i, label in enumerate(knots_names):
            years = knots_jyears[label]
            idx = (slider_date.val-1.5 < years) & (years < slider_date.val)

            if np.sum(idx) == 0:
                continue # we did not select any component from this component, next one

            x = np.array(knots_X[label])
            y = np.array(knots_Y[label])

            # get points iside current rectangle for current date
            rect_idx = (x1 < x) & ( x < x2) & (y1 < y) & ( y < y2) & idx 

            if np.sum(rect_idx) > 0:
                textbox.set_val(label)
                selected_knot = label
                selected_x = x[rect_idx].ravel()
                selected_y = y[rect_idx].ravel()
                selected_ind = knots[label].index[rect_idx]
                log.debug(f'Selected {label} points  rect_idx {rect_idx} x {x[rect_idx]}, y {y[rect_idx]} with indices {selected_ind}')
                textbox.begin_typing(None)
                break # if we find selected components in this epoch, continue with renaming
            else:
                pass

        update(slider_date.val)


    def toggle_selector(event):
        log.debug('GUI: Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            log.debug('Selector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['S', 's'] and not toggle_selector.RS.active:
            log.debug('Selector activated.')
            toggle_selector.RS.set_active(True)
        if event.key in ['R', 'r']:
            log.debug('Selector deactivated.')
            toggle_selector.RS.set_active(False)
            textbox.begin_typing(None)
            #textbox.set_val('')


    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=0, minspany=0,
                                           spancoords='data',
                                           interactive=False)


    #plt.connect('key_press_event', toggle_selector)
    fig.canvas.mpl_connect('key_press_event', toggle_selector)



    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider_slider = make_axes_locatable(ax)
    slider_ax = divider_slider.append_axes("top", size="3%", pad="4%")  
    slider_date = Slider(ax=slider_ax, label="Date", valmin=2007, valmax=2020, valinit=2008, valstep=0.2, orientation="horizontal")
    slider_date.on_changed(update)

    #divider_textbox = make_axes_locatable(ax)
    #textbox_ax = divider_textbox.append_axes("bottom", size="3%", pad="4%") 
    textbox_ax = fig.add_axes([0.3,0.015,0.5,0.05])
    textbox = TextBox(textbox_ax, 'Knot name:', initial='None')
    textbox.on_submit(submit_textbox)



    ax.set_xlim([-1.0, +1.0])
    ax.set_ylim([-1.0, +1.0])
    ax.set_aspect('equal')

    fig.suptitle('S to select, R to rename, Q to deactivate selector')

    print('S to select, R to rename, Q to deactivate selector')
    print('(you can select points from one component at a time)')
    print('(if you use the zoom or movement tools, remember to unselect them)')

    plt.show()
    
    return mod






def KnotsIdGUI(mod):
    """
        Prompt a GUI to select identified knots and alter their label, reprenting their
        time evolution.
        
        It can be used inside jupyter notebooks, using '%matplotlib widget' first.
       
        Parameters:
        -----------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot, with at least columns 
             'label', 'date', 'X', 'Y', 'Flux (Jy)'.
        
        Returns:
        --------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot with their altered labels, with 
             at least columns 'label', 'date', 'X', 'Y', 'Flux (Jy)'.
    """
    
    mod = mod.copy()
    
    knots = dict(tuple(mod.groupby('label')))
    knots_names = list(knots.keys())
    knots_values = list(knots.values())
    knots_jyears = {k:Time(knots[k]['date'].to_numpy()).jyear for k in knots}
    knots_dates = {k:knots[k]['date'].to_numpy() for k in knots}
    knots_fluxes = {k:knots[k]['Flux (Jy)'].to_numpy() for k in knots}

    from matplotlib.widgets import TextBox, RectangleSelector

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
    
    lineas = list()
    textos = list()
    
    def draw_all():
        nonlocal lineas, textos

        for linea in lineas:
            if linea is not None:
                linea.remove()
        for texto in textos:
            if texto is not None:
                texto.remove()

        ax.set_prop_cycle(None)
        
        lineas = list()
        textos = list()
        
        #ax.clear() # either clear the whole axis or remove every artist separetly
        
        for i, label in enumerate(knots_names):
            x, y = knots_jyears[label], knots_fluxes[label]

            if len(x) > 0:
                lineas.append(ax.plot(x, y, '.-', linewidth=0.8, alpha=0.5, label=label)[0])
            else:
                lineas.append(None)
                
            if len(x) > 0:
                #textos.append(ax.annotate(label, (x[0], y[0]), (-28,-10), textcoords='offset points', color=lineas[i].get_color(), fontsize=14))
                textos.append(ax.text(x[0], y[0], label, {'color':lineas[i].get_color(), 'fontsize':14}))
            else:
                textos.append(None)
                
        fig.canvas.draw_idle() # if removed every artist separately instead of ax.clear()


    def update():
        nonlocal lineas, textos

        for i, label in enumerate(knots_names):   
            x, y = knots_jyears[label], knots_fluxes[label]

            if lineas[i] is not None:
                if len(x) > 0:
                    lineas[i].set_xdata(x)
                    lineas[i].set_ydata(y)
                else:
                    lineas[i].remove()
                    lineas[i] = None
            else:
                if len(x) > 0:    
                    lineas[i] = ax.plot(x, y, '.-', linewidth=0.8, alpha=0.5, label=label)[0]
        
            if textos[i] is not None:
                if len(x) > 0:    
                    textos[i].set_position((x[0], y[0]))
                    textos[i].set_text(label)
                else:
                    textos[i].remove()
                    textos[i] = None
                    #textos[i].set_position((10, 10))
            else:
                if len(x) > 0:    
                    #textos[i] = ax.annotate(label, (x[0], y[0]), (-24,-10), textcoords='offset points', color=lineas[i].get_color(), fontsize=15)
                    textos[i] = ax.text(x[0], y[0], label, {'color':lineas[i].get_color(), 'fontsize':14})
        
        fig.canvas.draw_idle()    
   

    selected_knot = None
    selected_ind = None
    selected_date = None
    selected_flux = None
    
    def submit_textbox(text):
        nonlocal mod, knots, knots_names, knots_values, knots_jyears, knots_dates, knots_fluxes

        log.debug('Submited with:')
        log.debug(f'   selected_knot {selected_knot}')
        log.debug(f'   selected_ind {selected_ind}')
        log.debug(f'   selected_flux {selected_flux}')
        log.debug(f'   selected_date {selected_date}')

        if selected_knot is not None:        
            mod.loc[selected_ind, 'label'] = text.upper()

            knots = dict(tuple(mod.groupby('label')))
            knots_names = list(knots.keys())
            knots_values = list(knots.values())
            knots_jyears = {k:Time(knots[k]['date'].to_numpy()).jyear for k in knots}
            knots_dates = {k:knots[k]['date'].to_numpy() for k in knots}
            knots_fluxes = {k:knots[k]['Flux (Jy)'].to_numpy() for k in knots}

            print(f"Updated index {selected_ind} to {text.upper()}")
        else:
            pass
        
        draw_all()

    def line_select_callback(eclick, erelease):
        nonlocal selected_knot,selected_date, selected_flux, selected_ind

        # 1 eclick and erelease are the press and release events
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        log.debug("GUI: (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        log.debug("GUI:  The button you used were: %s %s" % (eclick.button, erelease.button))

        selected_knot = None
        selected_date = None
        selected_flux = None
        selected_ind = None

        for i, label in enumerate(knots_names):
            years = knots_jyears[label]
            fluxes = knots_fluxes[label]
            
            # get points iside current rectangle for current date
            rect_idx = (x1 < years) & ( years < x2) & (y1 < fluxes) & ( fluxes < y2) 
            
            if np.sum(rect_idx) == 0:
                continue # we did not select any component from this component, next one

            x = np.array(years)
            y = np.array(fluxes)

            if np.sum(rect_idx) > 0:
                textbox.set_val(label)
                selected_knot = label
                selected_x = x[rect_idx].ravel()
                selected_y = y[rect_idx].ravel()
                selected_ind = knots[label].index[rect_idx]
                log.debug(f'Selected {label} points  rect_idx {rect_idx} date {x[rect_idx]}, flux {y[rect_idx]} with indices {selected_ind}')
                break # if we find selected components in this epoch, continue with renaming
            else:
                pass

        update()


    def toggle_selector(event):
        log.debug('GUI: Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            log.debug('Selector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['S', 's'] and not toggle_selector.RS.active:
            log.debug('Selector activated.')
            toggle_selector.RS.set_active(True)
        if event.key in ['R', 'r']:
            log.debug('Selector deactivated.')
            toggle_selector.RS.set_active(False)
            textbox.begin_typing(None)
            #textbox.set_val('')


    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=0, minspany=0,
                                           spancoords='data',
                                           interactive=False)


    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider_textbox = make_axes_locatable(ax)
    textbox_ax = divider_textbox.append_axes("bottom", size="10%", pad="15%") 
    #textbox_ax = fig.add_axes([0.3,0,0.5,0.05])
    textbox = TextBox(textbox_ax, 'Knot name:', initial='None')
    textbox.on_submit(submit_textbox)
    
    draw_all()
    
    #ax.autoscale()
    ax.set_xlabel('date (year)')
    ax.set_ylabel('Flux (Jy)')
    ax.set_title('Flux from each component')
    
    xlims = Time(np.amin(mod['date'])).jyear, Time(np.amax(mod['date'])).jyear
    ax.set_xlim((xlims[0]-0.03*np.abs(xlims[1]-xlims[0]), xlims[1]+0.03*np.abs(xlims[1]-xlims[0])))
    plt.show()

    return mod



def KnotsIdReadMod(path=None, file_list=None):
    """
        Read *_mod.mod files as printed by diffmap, return a dataframe containing all information ready
        to be worked on for labelling and to be used with these GUIs.
   
        Parameters:
        -----------
         path : :str:
             string indicating the path to the mod files to be used, their names must end in the format
             '%Y-%m-%d_mod.mod', for example, path = 'vlbi/ftree/*/*_mod.mod'.
             
        Returns:
        -----------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot, with at least columns 
             'label', 'date', 'X', 'Y', 'Flux (Jy)'.
    """
    
    mod_dates_str = list()
    mod_dates = list()
    mod_data = list()

    for f in glob.glob(f'{path}'):
        match = re.findall(r'([0-9]{4}-[0-9]{2}-[0-9]{2})_mod.mod', f)
        if not len(match) > 0:
            continue

        date_str = match[0]
        date = datetime.strptime(date_str, '%Y-%m-%d')

        mod_dates_str.append(date_str)
        mod_dates.append(date)
        mod_data.append(pd.read_csv(f, sep='\s+', comment='!', names=['Flux (Jy)', 'Radius (mas)', 'Theta (deg)', 'Major FWHM (mas)', 'Axial ratio', 'Phi (deg)', 'T', 'Freq (Hz)', 'SpecIndex']))

    # sort by date    
    idx = np.argsort(mod_dates)
    mod_dates_str = list(np.array(mod_dates_str, dtype=object)[idx])
    mod_dates = list(np.array(mod_dates, dtype=object)[idx])
    mod_data = list(np.array(mod_data, dtype=object)[idx])

    # fix stupid 'v' in columns, insert a label field, add X, Y columns
    for i in range(len(mod_dates)):
        mod_data[i].insert(0, 'label', value=None)
        mod_data[i].insert(1, 'date', value=mod_dates[i])

        mod_data[i]['Flux (Jy)'] = mod_data[i]['Flux (Jy)'].str.strip('v').astype(float)
        mod_data[i]['Radius (mas)'] = mod_data[i]['Radius (mas)'].str.strip('v').astype(float)
        mod_data[i]['Theta (deg)'] = mod_data[i]['Theta (deg)'].str.strip('v').astype(float)

        mod_data[i].insert(5, 'X', mod_data[i]['Radius (mas)']*np.cos(np.pi/180*(mod_data[i]['Theta (deg)']-90)))
        mod_data[i].insert(6, 'Y', mod_data[i]['Radius (mas)']*np.sin(np.pi/180*(mod_data[i]['Theta (deg)']-90)))

     
    mod = pd.concat(mod_data, ignore_index=True)
    
    return mod


def KnotsIdSaveMod(mod, path=None):
    pass

def KnotsIdReadCSV(path=None):
    """
        Read Knots data to .csv files (as done by Svetlana? ##)
        
        Each knot label has its own {label}.csv. To be compatible with Svetlana's format, 
        columns should be modified to:
        'Date' (jyear), 'MJD', 'X(mas)', 'Y(mas)', 'Flux(Jy)'
        These columns are derived from the ones in `mod`, old ones are removed.
        
        Parameters:
        -----------
         path: :str:
             string containing the path to read files from eg:
             path = 'myknows/*.csv'
        Returns:
        --------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot, with at least columns 
             'label', 'date', 'X', 'Y', 'Flux (Jy)'.
    """
    
    if path is None:
        log.error('Path not specified')
        raise Exception('Path not specified')
    
    dataL = list()
    
    for f in glob.glob(f'{path}'):
        match = re.findall(r'/(.*).csv', f)
        
        if not len(match) > 0:
            continue
        
        knot_name = match[0]
        
        log.debug(f'Loading {knot_name} from {f}')
        
        knot_data = pd.read_csv(f, parse_dates=['date'], date_parser=pd.to_datetime)
        
        dataL.append((knot_name, knot_data))
    
    mod = pd.concat(dict(dataL), ignore_index=True)

    return mod



def KnotsIdSaveCSV(mod, path=None):
    """
        Save Knots data to .csv files (as done by Svetlana? ##)
        
        Each knot label has its own {label}.csv. To be compatible with Svetlana's format, 
        columns should be modified to:
        'Date' (jyear), 'MJD', 'X(mas)', 'Y(mas)', 'Flux(Jy)'
        These columns are derived from the ones in `mod`, old ones are removed.
        
        Parameters:
        -----------
         mod : :pd.DataFrame:
             pandas.DataFrame containing every knot, with at least columns 
             'label', 'date', 'X', 'Y', 'Flux (Jy)'.
         path: :str:
             string containing the path to which the files are to be saved, eg:
             path = 'my_knots/'
             
        Returns:
        --------
         None
    """
    
    if path is None:
        log.error('Path not specified')
        raise Exception('Path not specified')
        
        
    mod = mod.copy()
    
    
    mod_dict = dict(list(mod.groupby('label')))
    
    for label, data in mod_dict.items():
        data = data.copy()
        
        #data.insert(0, 'Date', Time(data['date']).jyear)
        #data.insert(1, 'MJD', Time(data['date']).mjd)
        #data = data.rename(columns={'X': 'X (mas)', 'Y': 'Y (mas)'})
        #data = data.drop(columns=['date'])
        #data = data.drop(columns=['label'])
        #if 'Radius (mas)' in data.columns:
        #    data = data.rename(columns={'Radius (mas)':'R(mas)'})
        #data.columns = data.columns.str.replace(' ', '')
        
        data.to_csv(f'{path}/{label}.csv', index=False) 