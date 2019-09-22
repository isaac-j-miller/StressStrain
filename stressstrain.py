import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

count = 0
MATERIAL_NAME_PORTIONS = \
    {
        'pol': 'Polypropylene',
        'hdpe': 'High-Density Polyethylene',
        'bir': 'Birch Wood',
        'cop': 'Copper',
        'low': 'Low-Carbon Steel',
        'acr': 'Acrylic (amorphous)',
        'oak': 'Oak Wood',
        'nyl': 'Nylon (semi-crystalline)',
        '2024': '2024 Aluminum',
        'map': 'Maple Wood',
        '1080': '1080 Spring Steel',
        '304': '304 Stainless Steel'

    }

MATERIAL_LER_THRESHOLDS = \
    {
        'metal': .01,
        'plastic': .01,
        'wood': .03,
        'nylon': 0,
        'unknown': .03
    }

PALLETS = \
    {
        'polypropylene': 'Blues_r',
        'polyethylene': 'BuGn_r',
        'acrylic': 'RdPu',
        'birch': 'GnBu_r',
        'maple': 'PuBuGn_r',
        'oak': 'BuPu_r',
        'carbon': 'Greys_r',
        '1080': 'Greens_r',
        '304': 'OrRd_r',
        '2024': 'Purples_r',
        'copper': 'YlOrBr_r',
        'nylon': 'Greys_r'
    }


def splitStupidLine(line, cols):
    out = []
    origline = line
    if line.count('"') !=0:
        for i in range(cols):
            try:
                start = line.index('"')
                line = line[start+1:]
                end = line.index('"')
                out.append(line[:end].replace(',',''))
                line = line[end+1:]
            except ValueError as a:
                print(origline, '\n', line, len(origline))
                raise Exception(a)
    else:
        out = ['','','']
    return ','.join(out) + '\n'


class Diagram:
    plt.ioff()
    pl=sns.lineplot([0],[0])
    pl.legend().set(zorder=101)
    plotName = ''

    ysmarker = Line2D([0], [0], color='black', marker='.', label='Yield Point', lw=0)
    usmarker = Line2D([0], [0], color='black', marker='^', label='Ultimate Point', lw=0)
    fsmarker = Line2D([0], [0], color='black', marker='x', label='Fracture Point', lw=0)
    elems = [ysmarker]
    trial = \
        {
            'polypropylene': 0,
            'polyethylene': 0,
            'birch': 0,
            'acrylic': 0,
            'maple': 0,
            'oak': 0,
            'carbon': 0,
            '1080': 0,
            '304': 0,
            '2024': 0,
            'copper': 0,
            'nylon': 0,
        }

    def __init__(self, source, name, count = 0):
        self.count = count
        print('name:', name, count)

        self._name = name
        head = {}
        lines = None
        lim = 0
        temp = 'temp.csv'
        with open(source, 'r') as f:
            lines = f.readlines()
            lim = lines.index('\n')
            self._head = {line.split(',')[0]:line.split(',')[1].replace('\n','').replace('"','') for line in lines[:lim]}
        with open(temp, 'w') as f:
            f.write('\n'.join([splitStupidLine(line, 3) for line in lines[lim+3:]]))
        #print(head)
        self._clear_yield = None
        self._shape = self._head['Geometry']
        self._length = float(self._head['Length'])
        if self._shape == 'Rectangular':
            self._area = float(self._head['Thickness'])*float(self._head['Width'])
        else:
            self._area = np.pi * (float(self._head['Diameter']) / 2) ** 2

        self._data = pd.read_csv(temp, header=None, dtype=float)

        self._data = self._data.rename(axis = 1, mapper= {0: 'time',1: 'extension', 2:'load'})
        #self._data = self._data[self._data.index >= 5]
        #self._data = self._data.astype(float)
        self._volume = self._area*self._length
        self._data.extension -= self._data.extension.iloc[0]
        self._data.load -= self._data.load.iloc[0]
        self._data['stress'] = self._data['load']/self._area
        self._data['strain_percent'] = self._data['extension']*100/self._length
        self._data.drop(axis=1, labels=['time'], inplace=True)
        self._data['moving_avg'] = self._data.stress.ewm(adjust=False, span=5).mean()
        self._data['deriv1'] = self._data.moving_avg.diff()
        self._data['slopestd'] = self._data.deriv1.rolling(window=10).std().ewm(adjust=False,span=10).mean().diff()
        self._data['deriv2'] = self._data.deriv1.diff()
        self._fracture_strain = None
        self._ra_percent = None
        self._el_percent = None
        self._true_stress_failure = None
        self._true_strain_failure = None
        self._fracture_strength = None
        self._final_max = None
        self._end_ler = None
        self._yield_strength = None
        self._elastic_mod = None
        self._calculated_offset = False
        self._yield_point = (None, None)
        # determine material type
        low = self._name.lower()
        if 'steel' in low or 'aluminum' in low:
            self._type = 'metal'
        elif 'nylon' in low or 'density' in low or 'copper' in low:
            self._type = 'nylon'
        elif 'poly' in low or 'acrylic' in low:
            self._type = 'plastic'
        elif 'wood' in low:
            self._type = 'wood'
        else:
            self._type = 'unknown'
        self._colors = ''
        k = [key for key in PALLETS.keys() if key in self._name.lower()][0]
        self._colors = PALLETS[k]
        self._trial = Diagram.trial[k]
        Diagram.trial[k] += 1

    def trim(self, min_strain_percent, max_strain_percent):
        self._data = self._data[(self._data.strain_percent>=min_strain_percent) & (self._data.strain_percent <= max_strain_percent)]

    def determine_elastic_mod_end(self):
        try:
            end = self._data.strain_percent.iloc[10]
        except IndexError:
            thresh = MATERIAL_LER_THRESHOLDS[self._type]
            while True:
                try:
                    end = self._data[(self._data.slopestd > thresh) &
                                     (self._data.strain_percent > 0) &
                                     (self._data.slopestd > self._data.slopestd.shift(-1)) &
                                     (self._data.slopestd > self._data.slopestd.shift(1))].strain_percent.values[0]
                    break
                except IndexError:
                    end = self._data.strain_percent.iloc[5]
                    break

        self._end_ler = end
        return end

    def calc_elastic_mod(self, elastic_region_end=None):
        if elastic_region_end is None:
            elastic_region_end = self._end_ler

        elastic_point = self._data[self._data.strain_percent == elastic_region_end].stress.values[0]

        self._elastic_mod = (elastic_point/elastic_region_end).mean()
        return self._elastic_mod

    def calc_yield_strength(self):
        w = 20
        print('end elastic mod: ', self._end_ler)
        self._data['moving_avg2'] = self._data.stress.ewm(adjust=False, span=w).mean()
        maxima = self._data[(self._data.strain_percent > self._end_ler) &
                            (self._data.index < self._data[self._data.strain_percent == self._end_ler].index[0] + 500) &
                            ((self._data.moving_avg2.shift(-1)) < self._data.moving_avg2) &
                            ((self._data.moving_avg2.shift(w-1)) < self._data.moving_avg2) &
                            (self._data.deriv2 < 0)]

        if len(maxima) and self._type != 'nylon':
            decision = self._data.shift(w//4)[self._data.index == maxima[maxima.deriv2 == maxima.deriv2.min()].index[0]]

            self._yield_point = (decision.strain_percent.values[0], decision.stress.values[0])
            self._yield_strength = self._yield_point[1]
            return self._yield_strength
        else:
            return None

    def generate_offset_line(self):
        if self._elastic_mod is not None:
            self._data['offset_line'] = (self._data.strain_percent - 0.2)*self._elastic_mod
            self._calculated_offset = True
        else:
            print('please calc_elastic_mod() first')

    def calculate_intersection(self):
        try:
            if self._calculated_offset:
                left = self._data[(self._data.stress <= self._data.offset_line) &
                                  (self._data.stress.shift(1) >= self._data.offset_line.shift(1))].index[0]
                right = left + 1

                # prepare for interpolation
                x1 = self._data[self._data.index == left].strain_percent.values[0]
                x2 = self._data[self._data.index == right].strain_percent.values[0]

                y1a = self._data[self._data.index == left].stress.values[0]
                y1b = self._data[self._data.index == left].offset_line.values[0]

                y2a = self._data[self._data.index == right].stress.values[0]
                y2b = self._data[self._data.index == right].offset_line.values[0]

                ma = (y2a-y1a)/(x2-x1)
                mb = (y2b-y1b)/(x2-x1)

                x = (y1b-y1a)/(ma-mb) + x1
                y = (y2a - y1a)*(x - x1)/(x2-x1) + y1a

                self._yield_point = (x, y)
                self._yield_strength = y

        except IndexError:
            print(self._name + ': ERROR - 0.2% OFFSET LINE DOES NOT INTERSECT CURVE')

            return
        return self._yield_point

    def calc_ultimate_strength(self):
        self._final_max = self._data.stress.max()
        if self._fracture_strain is not None:
            if self._final_max == self._data[self._data.strain_percent == self._fracture_strain].stress.values[0]:
                self._final_max = None
        elif self._final_max in self._data.stress.values[-5:]:
                self._final_max = None
        return self._final_max

    def plot(self, save=False, display = True, showoffset=True, showtext = True):
        if not display:
            plt.ioff()
        else:
            plt.ion()

        with sns.color_palette(self._colors):
            sns.lineplot(self._data['strain_percent'], self._data['stress'], color= sns.color_palette(self._colors)[self._trial])
            if self._calculated_offset and showoffset:
                sns.lineplot(self._data['strain_percent'], self._data['offset_line'])

            elif self._yield_strength is None:
                print('please generate_offset_line() first')
            if self._yield_strength is not None:
                if showtext:
                    Diagram.pl.text(self._yield_point[0], self._yield_point[1], 'yield point', horizontalalignment='right')

                Diagram.pl.plot(self._yield_point[0], self._yield_point[1], marker = '.', color = 'black', zorder=100)
                #pl.plot(self._end_ler, self._data[self._data.strain_percent == self._end_ler].stress.values[0], marker='o') test end of ler
            if self._final_max is not None:
                ultimate_strain = self._data[self._data.stress == self._final_max].strain_percent.values[0]
                if showtext:
                    Diagram.pl.text(ultimate_strain, self._final_max, 'ultimate tensile strength', horizontalalignment='right')
                Diagram.pl.plot(ultimate_strain, self._final_max, marker = '^', color = 'black', zorder=100)
            if self._fracture_strength is not None:
                if showtext:
                    Diagram.pl.text(self._fracture_strain, self._fracture_strength, 'fracture strength', horizontalalignment='right')
                Diagram.pl.plot(self._fracture_strain, self._fracture_strength, marker = 'x', color = 'black', zorder=100)
            #pl.axvline(self._end_ler, 0, self._data.stress.max()*1.10)
            Diagram.pl.set(xlabel='Strain (%in/in)', ylabel='Stress (psi)', title=self._name if Diagram.plotName =='' else Diagram.plotName,
                   #ylim=(0, self._data.stress.max() * 1.10),
                   #xlim=(0, self._data.strain_percent.max()*1.10)
                           )

            telems = []
            if self._final_max is not None:
                telems.append(Diagram.usmarker)
            if self._fracture_strength is not None:
                telems.append(Diagram.fsmarker)
            for item in telems:
                if item not in Diagram.elems:
                    Diagram.elems.append(item)
            existingNames = [item.get_label() for item in Diagram.elems]
            print(existingNames)
            print(self._name)
            if self._name not in existingNames:
                Diagram.elems.append(Line2D([0], [0], color=sns.color_palette(self._colors)[self._trial], label=self._name))
        delems = [item.get_label().lower() for item in Diagram.elems]
        if 'ultimate point' in delems:
            index = delems.index('ultimate point')
            if index!= 1:
                copy = Diagram.elems[1]
                Diagram.elems[1] = Diagram.usmarker
                Diagram.elems[index] = copy
        if 'fracture point' in delems:
            index = delems.index('fracture point')
            if index != 2:
                copy = Diagram.elems[2]
                Diagram.elems[2] = Diagram.fsmarker
                Diagram.elems[index] = copy

        Diagram.pl.legend(handles = Diagram.elems, fancybox=True, loc=0)

        plt.tight_layout()
        if save:
            plt.savefig(self._name+str(self._trial) + '-plot-'+str(self.count+1)+'.png')
            self.count += 1

    def print_info(self, suppress = False):
        lines = ['figure: '+ self._name+str(self._trial) + '-plot-'+str(self.count)+'.png']
        if self._shape == 'Rectangular':
            lines.append('shape: rectangular')
            lines.append('width: {w:.6f} in'.format(w=float(self._head['Width'])))
            lines.append('thickness: {w:.6f} in'.format(w=float(self._head['Thickness'])))
        else:
            lines.append('shape: cylindrical')
            lines.append('diameter: {w:.6f} in'.format(w=float(self._head['Diameter'])))
        lines.append('length: {w:.6f} in'.format(w=self._length))

        if self._elastic_mod is not None:
            lines.append( ''.join(['elastic modulus: {emod:.2f} '.format(emod = self._elastic_mod/(1000 if self._elastic_mod > 1000 else 1)),
                  'ksi' if self._elastic_mod>1000 else 'psi']))
            lines[-1] += ', determined using the tangent modulus method'
        if self._yield_strength is not None:
            lines.append( ''.join(['yield strength: {ystr:.2f} '.format(ystr = self._yield_strength/(1000 if self._yield_strength > 1000 else 1)),
                  'ksi' if self._yield_strength>1000 else 'psi']))
            if self._clear_yield is not None:
                lines.append ('clear yield point at ({strain:.4f},{stress:.4f})'.format(strain = self._yield_point[0], stress=self._yield_point[1])
                     if self._clear_yield else 'no clear yield point. 0.2% offset method used')
        if self._final_max is not None:
            lines.append(''.join(['ultimate strength: {fstr:.2f} '.format(fstr = self._final_max/(1000 if self._final_max > 1000 else 1)),
                  'ksi' if self._final_max>1000 else 'psi']))
        else:
            lines.append('ultimate strength not reached')
        if self._fracture_strength is not None:
            lines.append(''.join(['fracture strength: {fstr:.2f} '.format(fstr = self._fracture_strength/(1000 if self._fracture_strength > 1000 else 1)),
                  'ksi' if self._fracture_strength>1000 else 'psi']))

            lines.append('true strain at failure: {fstr:.2f} (in/in)'.format(fstr=self._true_strain_failure))
            lines.append(''.join(['true stress at failure: {fstr:.2f} '.format(
                fstr=self._true_stress_failure / (1000 if self._true_stress_failure > 1000 else 1)),
                                  'ksi' if self._true_stress_failure > 1000 else 'psi']))
            lines.append('reduction in area: {fstr:.2f}%'.format(fstr=self._ra_percent ))
            lines.append('elongation: {fstr:.2f}%'.format(fstr=self._el_percent))
        else:
            lines.append('fracture point not reached')

        output = [self._name.title() + ', trial ' + str(self._trial) +', count: '+ str(self.count) + ' Info: ', *lines]

        if not suppress:
            for line in output:
                print(line)

        return output

    def calculate_fracture_strength(self):

        change = None
        tail = self._data.iloc[-150:]
        change = tail[tail.deriv1 < -20].index
        i = 0
        while len(change) == 0:
            change = tail[tail.deriv1 < -20 + i].index

            i += 1
        maxdelta = change = tail[tail.deriv1 < -20 + i].slopestd.max()
        print(i)
        if i < 10:
            index = tail[tail.slopestd==maxdelta].index[0]
            line = self._data[self._data.index == index -10]
            self._fracture_strength = line.stress.values[0]
            self._fracture_strain = line.strain_percent.values[0]
            if self._fracture_strength != self._final_max:  # if fracture
                self._true_strain_failure = np.log(1 + line.strain_percent.values[0]/100)
                self._el_percent = 100 * line.extension.values[0]/self._length
                newArea = self._volume / (line.extension.values[0] + self._length)
                self._ra_percent = 100*(self._area - newArea)/self._area
                self._true_stress_failure = line.load.values[0]/newArea
            else:
                self._fracture_strength = None
        else:
            self._fracture_strength = None


        return change

    def process(self):

        # first, trim the end stuff off
        change = self.calculate_fracture_strength()

        self.calc_ultimate_strength()
        #if self._final_max is not None:
        #    self.trim(0, self._data[self._data.index == change[0]].strain_percent.values[0])

        # then calc elastic mod
        self.determine_elastic_mod_end()
        if self._end_ler is not None:
            self.calc_elastic_mod(self._end_ler)
        # then check if there is a clear yield point
        if self.calc_yield_strength() is None:  # if no clear yield point
            #print('no clear yield point')
            self._clear_yield = False
            # if not, generate offset line
            self.generate_offset_line()
            # calc yield strength
            self.calculate_intersection()
        else:
            self._clear_yield = True
            #print('clear yield point at',self._yield_point)
        if self._fracture_strength is not None:
            self.trim(0, self._fracture_strain)
        # print info
        #self.print_info()
        # plot
        #self.plot()

    def rename(self, name):
        self._name = name


def folderwalk(home, keyphrase =[''], concat=False):

    folders = [f for f in os.listdir(home) if any([key in f.lower() for key in keyphrase])]

    with open('data.txt', 'w') as d:
        data = '\n\n'.join([processFolder(home, f, concat) for f in folders])
        d.writelines(data)


def processFolder(home, folder, concat=False):
    global count
    basepath = os.path.join(home, folder)
    print(basepath)
    files = os.listdir(basepath)
    print(folder)
    name = ''
    nameKey = [k for k in MATERIAL_NAME_PORTIONS.keys() if k in folder.lower()]
    if len(nameKey) == 1:
        name = MATERIAL_NAME_PORTIONS[nameKey[0]]
    else:
        print('keys: ', nameKey)
        name = input('Please name this data. The folder is ' + folder)


    dat = []
    for file in files:
        if not concat:
            plt.close()
        src = os.path.join(basepath,file)
        print(src)
        diag = Diagram(src,name,count=count)
        try:
            diag.process()
            diag.plot(True, False, False, False)
            dat.append(os.path.join(folder, file) + '\n '+'\n'.join(diag.print_info(True)))
        except Exception as a:
            if 'DATA_ERROR' in str(a):
                dat.append(os.path.join(folder, file) + '; ' + name + ': ERROR - file is shit')
            else:
                raise Exception(a)

        count+=1
    return '\n#####\n'.join(dat)


if __name__=='__main__':
    Diagram.plotName = 'Woods'
    folderwalk('C:/Users/Isaac Miller/Documents/GitHub/StressStrain/materials_lab1', ['wood'], True)
    #folder = 'C:/Users/Isaac Miller/Documents/GitHub/StressStrain/materials_lab1/ENGR314_FA19_0001_1080 Spring Steel_B_20190903_JAZ_1.is_tens_RawData/Specimen_RawData_2.csv'
    #d = Diagram(folder, '1080 Spring Steel')
    #d.plot()
    plt.show()



