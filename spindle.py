# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import warnings
import os
import report_generator as rg

class Signal (object):

    def __init__(self, path, Fs):
        '''
        :param path: path to file with signal
        :param Fs:
        '''
        self.path = path
        self.s = np.fromfile(path, dtype='float32')
        self.T = len(self.s)/Fs
        self.Fs = Fs
        self.nyq_f = Fs / 2
        self.t = np.arange(0, self.T, 1 / Fs)
        self.window = Fs * 20

    def plot_signal(self, show = True):
        '''
        :param show:
        :return:
        '''
        plt.figure(figsize=((20, 10)))
        plt.plot(self.t[:self.window], self.s[:self.window])
        plt.title("Signal Windowed 0s - {}s".format(int(self.window/self.Fs)))
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Value")
        plt.savefig("./report/images/signal.png")
        if show: plt.show()

    def filt_classifier(self, verbose = True):
        '''
        :param verbose:
        :return:
        '''

        f_page = rg.Page("./report/filt.html", u"Klasyfikator Wrzecion Snu", u"Metoda Filtrowania", u"Paweł Pierzchlewicz")
        f_page.image('./images/signal.png', u"Okienkowany Sygnał", u"Przedstawienie sygnału okienkowanego prostokątnie na przedziale 0s - {}s.\n Reprezentuje jedną składkę EEG.".format(int(self.window/self.Fs)))

        # FIRST FILTERING
        N = 7
        Wn = (11 / self.nyq_f, 16 / self.nyq_f)
        [b, a] = ss.butter(N, Wn, btype='bandpass')
        s_1 = ss.filtfilt(b, a, self.s)
        s_1 *= s_1

        # SHOW SIGNAL AFTER FILTERING IF VERBOSE
        if verbose:
            plt.figure(figsize=((20, 10)))
            plt.plot(self.t[:self.window], s_1[:self.window])
            plt.title("Signal after bandpass 11 - 16Hz\nWindowed: 0s - {}s".format(int(self.window/self.Fs)))
            plt.xlabel("Time (s)")
            plt.savefig("./report/images/filt1.png")
            #plt.show()
            f_page.image('./images/filt1.png', u"Przeflitrowany sygnał I",
                         u"Przedstawienie sygnału okienkowanego prostokątnie na przedziale 0s - {}s. Sygnał został przefiltrowany filtrem pasmowym z częstościami odcięcia 11 Hz i 16 Hz i następnie podniesiony do kwadratu".format(
                             int(self.window / self.Fs)))

        # SECOND FILTERING
        N = 7
        Wn = 10 / self.nyq_f
        [b, a] = ss.butter(N, Wn, btype='lowpass')
        s_2 = ss.filtfilt(b, a, s_1)
        s_2 = s_2 ** .5

        # SHOW SIGNAL AFTER FILTERING IF VERBOSE
        if verbose:
            plt.figure(figsize=((20, 10)))
            plt.plot(self.t[:self.window], s_2[:self.window])
            plt.title("Signal after bandpass 11 - 16Hz and lowpass 10Hz\nWindowed: 0s - {}s".format(int(self.window / self.Fs)))
            plt.xlabel("Time (s)")
            plt.savefig("./report/images/filt2.png")
            #plt.show()
            f_page.image('./images/filt2.png', u"Przeflitrowany sygnał II",
                         u"Przedstawienie sygnału okienkowanego prostokątnie na przedziale 0s - {}s. Sygnał został przefiltrowany filtrem dolnoprzepustowym z częstościami odcięcia 10 Hz, następnie wyciągnięto z niego pierwiastek".format(
                             int(self.window / self.Fs)))

        # FIND ALL ABOVE 5 mV
        s_2[s_2 < 5] = 0
        s_2[s_2 >= 5] = 1

        # SHOW SIGNAL AFTER BINARY TRANSFORM IF VERBOSE
        if verbose:
            plt.figure(figsize=((20, 10)))
            plt.plot(self.t[:self.window], s_2[:self.window])
            plt.title("Signal after binary mapping 5 mV\nWindowed: 0s - {}s".format(int(self.window / self.Fs)))
            plt.xlabel("Time (s)")
            plt.savefig("./report/images/filt3.png")
            #plt.show()
            f_page.image('./images/filt3.png', u"Sygnał przemapowany 0/1",
                         u"Przedstawienie sygnału okienkowanego prostokątnie na przedziale 0s - {}s. Sygnał został przemapowany na wartości 0 i 1, gdzie 0 reprezentuje wartości mniejsze niż 5 mV, a 1 wartości większe niż 5 mV".format(
                             int(self.window / self.Fs)))

        min_t = 0.5
        min_elements = int(min_t * self.Fs)

        length = len(s_2) - min_elements
        spindle = []
        for i in range(0, length):
            total = np.sum(s_2[i:i + min_elements])
            ratio = total / min_elements
            if ratio == 1:
                spindle.append(i)

        prev = spindle[0]
        filt_spindles = [spindle[0]]
        for w in spindle:
            if w - prev > 1:
                filt_spindles.append(prev + min_elements - 1)
                filt_spindles.append(w)
            prev = w
        filt_spindles.append(spindle[-1] + min_elements - 1)
        filt_spindles = np.array(filt_spindles)
        shape = filt_spindles.shape[0]
        filt_spindles = filt_spindles.reshape(int(shape / 2), 2)
        if verbose:
            print("{:<10}||{:>10}".format("Start", "End"))
            print(22*"=")
            for i in filt_spindles:
                print("{:<10}||{:>10}".format(i[0], i[1]))
        # the array has at pointer 0 the start point and pointer 1 the end point
        # Transform to time
        filt_spindles = filt_spindles / self.Fs
        f_page.table('Start', 'Koniec', np.round(np.array(filt_spindles),2), 'Przedziały Wrzecion', 'Przedziały czasowe [s], w których klasyfikator znalazł wrzeciona snu. Czyli odciniki w których wartość przefiltrowanego sygnału przekroczyła 5 mV na ponad 0.5s, w sumie znaleziono ' + str(len(filt_spindles)) + ' wrzeciona snu tą metodą.')
        f_page.pagination('./mp.html')
        f_page.close()
        return np.array(filt_spindles)  # [t_start, t_end]

    def mp_classifier(self, max_iteration, energy_error, verbose = True):
        '''
        :param max_iteration:
        :param energy_error:
        :param verbose:
        :return:
        '''
        f_page = rg.Page("./report/mp.html", u"Klasyfikator Wrzecion Snu", u"Metoda MP",
                         u"Paweł Pierzchlewicz")
        f_page.image('./images/signal.png', u"Okienkowany Sygnał",
                     u"Przedstawienie sygnału okieknowanego prostokątnie na przedziale 0s - {}s.\n Reprezentuje jedną składkę EEG.".format(
                         int(self.window / self.Fs)))

        def parametry_atomu(book, atom):
            f_Hz = atom['params']['f'] * book.fs / 2  # przekodowujemy częstość atomu na częstość w Hz
            A = atom['params']['amplitude']  # odczytujemy amplitudę
            phase = atom['params']['phase']  # odczytujemy fazę
            t0 = atom['params']['t'] / book.fs  # przeliczamy pozycję atomu z wartości w próbkach na wartości w sek.
            scale = atom['params']['scale'] / book.fs  # szerokość atomu w timeie (w sek.)
            return f_Hz, A, phase, t0, scale

        def tfr_atomu(book, atom, N_czestosci):
            f_Hz, A, phase, t0, scale = parametry_atomu(book, atom)
            t = np.arange(0, book.epoch_s / book.fs, 1 / book.fs)
            f = np.linspace(0, Fs / 2, N_czestosci)
            rec_t = np.zeros((1, book.epoch_s))
            rec_f = np.zeros((N_czestosci, 1))
            rec_t[0, :] = np.exp(-np.pi * ((t - t0) / scale) ** 2)  # obliczamy obwiednię timeową -> dla f. Gabora jest to f. Gaussa
            rec_f[:, 0] = np.exp(-np.pi * ((f - f_Hz) * scale) ** 2)  # obliczamy obwiednię częstotliwościową -> dla f. Gabora jest to f. Gaussa
            tfr_atom = np.kron(rec_t,
                               rec_f)  # przemnażamy przez siebie obwiednie -> to jest reprezentacja time-częstość pojedynczego atomu
            tfr_atom /= np.sum(np.sum(tfr_atom))  # normalizujemy
            tfr_atom *= atom['params']['modulus'] ** 2  # skalujemy energię
            return t, f, tfr_atom

        def rekonstrukcja_atomu(book, atom):
            f_Hz, A, phase, t0, scale = parametry_atomu(book, atom)
            t = np.arange(0, book.epoch_s / book.fs, 1 / book.fs)
            rekonstrukcja = A * np.exp(-np.pi * ((t - t0) / scale) ** 2) * np.cos(
                2 * np.pi * f_Hz * (t - t0) + phase)  # obliczamy przebieg timeowy atomu (funkcja Gabora) i
            return t, rekonstrukcja

        def TFRPlot(TFR, t_mapy, f_mapy, sig, Fs=128, title=''):
            '''
            Funkcja do rysowania map time-częstość z sygnałem zaprezentowanym poniżej
            TFR - mapa time-częstość (time-freqyency representation
            t_mapy, f_mapy - wektory reprezentujące osie timeu i częstości
            sig - sygnał do wyrysowania pod mapą (np. ten, z którego powstała mapa)
            Fs - częstość próbkowania sygnału
            title - tytuł do wyświetlenia ponad mapą
            '''
            df = f_mapy[1] - f_mapy[0]
            dt = t_mapy[1] - t_mapy[0]
            t = t_mapy
            plt.figure(figsize=(20, 10))
            sygAxes = plt.axes([0.05, 0.05, 0.8, 0.1])
            tfAxes = plt.axes([0.05, 0.15, 0.8, 0.8])
            sygAxes.plot(t, sig)
            plt.xlim((t_mapy.min(), t_mapy.max()))
            tfAxes.imshow(TFR, aspect='auto', origin='lower', interpolation='nearest',
                          extent=(
                          t_mapy.min() - dt / 2, t_mapy.max() + dt / 2, f_mapy.min() - df / 2, f_mapy.max() + df / 2))
            plt.setp(tfAxes, xticklabels=[])
            plt.title(title)
            plt.savefig("./report/images/{}.png".format(title.strip()))
            #plt.show()
            plt.clf()
            if np.sum(TFR):
                f_page.image("./images/{}.png".format(title.strip()), "Epoka " + title.split(' ')[-1],
                         u"Przedstawienie sygnału w przestrzeni czas częstość na przedziale {}s - {}s. Odrzucono wszystkie atomy, które nie zawierały się w paśmie 11Hz - 16Hz".format(str(np.round(t_mapy.min())), str(np.round(t_mapy.max()))))

        signal_file = self.path
        config_file = './mp_classify_config.set'
        Fs = self.Fs
        probes_per_epoch = self.Fs * 20 
        no_channels = 1
        selected_channels = 1  # wybrane kanały:
        #     a) numerowane są od 1
        #     b) składnia wybierania: 1, 3, 5, 7-11, 19
        no_epochs = int(len(self.s)/probes_per_epoch)
        selected_epochs = "1-{}".format(no_epochs)  # które epoki analizować, składnia wyboru tak jak dla kanałów

        maxIteracji = max_iteration  # liczba iteracji do wykonania dla jednej epoki, chyba,
        # że wcześniej zostanie osiągnięty zadany poniżej

        procentEnergii = 95.  # procent wyjaśnionej energii
        energyError = energy_error  # parametr regulujący gęstość słownika float w zakresie 0-1.
        # odpowiada minimalnej odległości między atomami słownika mierzonej w metryce iloczynów scalernych
        # nierówności (7) w cytowanym art.
        # Czym mniejszy tym większy słownik i tym dokładniejsza dekompozycja
        algorytm = 'SMP'  # możliwe wartości SMP | MMP1 |MMP2 |MMP3 - algorytmy opisane w art.

        # powyższe ustawiena zapisujemy do pliku tekstowego:
        fo = open(config_file, "wt")
        fo.write('# OBLIGATORY PARAMETERS\n')
        fo.write('nameOfDataFile  ' + signal_file + '\n')
        fo.write('nameOfOutputDirectory  ./\n')
        fo.write('writingMode            CREATE \n')  #
        fo.write('samplingFrequency      ' + str(Fs) + '\n')
        fo.write('numberOfChannels       ' + str(no_channels) + '\n')
        fo.write('selectedChannels       ' + str(selected_channels) + '\n')
        fo.write('numberOfSamplesInEpoch ' + str(probes_per_epoch) + '\n')
        fo.write('selectedEpochs         ' + str(selected_epochs) + '\n')
        fo.write('typeOfDictionary       OCTAVE_FIXED\n')
        fo.write('energyError            ' + str(energyError) + ' 100.0 \n')
        fo.write('randomSeed             auto \n')
        fo.write('reinitDictionary       NO_REINIT_AT_ALL \n')
        fo.write('maximalNumberOfIterations ' + str(maxIteracji) + '\n')
        fo.write('energyPercent             ' + str(procentEnergii) + '\n')
        fo.write('MP                        ' + algorytm + '\n')
        fo.write('scaleToPeriodFactor       1.0 \n')
        fo.write('pointsPerMicrovolt        1.0 \n')

        fo.write('\n# ADDITIONAL PARAMETERS\n')
        fo.write('normType                  L2 \n')
        fo.write(
            'diracInDictionary         YES \n')  # ta i poniższe linie włączają odpowiednie typy funkcji do słownika
        fo.write('gaussInDictionary         YES \n')
        fo.write('sinCosInDictionary        YES \n')
        fo.write('gaborInDictionary         YES \n')
        fo.close()

        bookName = self.path.split('.bin')[0] + '_smp.b'
        print(os.path.isfile(self.path.split('.bin')[0]+'_smp.b'))
        rerun = input("{} already exists, do you want to re run MP? [Y/N]".format(bookName))
        if rerun.lower() == "y":
            print("Rerunning MP")
            status = os.system('./empi-osx64 ' + config_file)
            if status == 32512:
                raise Exception("Cannot find empi executable, status code: {}".format(status))
            elif status != 0:
                raise Exception("Error at MP, status code: {}".format(status))
        else:
            print("Calculating Atoms")
        try:
            from book_reader import BookImporter
        except:
            warnings.warn('MISSING BOOK READER') # Throw Error
        bookName = self.path.split('.bin')[0]+'_smp.b'
        book = BookImporter(bookName)
        N_czestosci = int(book.fs)
        epoki = range(1, 31)  # Posłuży on nam do odtwarzania poszczególnych atomów
        total = 0
        mp_spindles = []
        for numerEpoki in epoki:
            rekonstrukcja = np.zeros(book.epoch_s)  # Przygotowujemy tablicę, w której powstanie rekonstrukcja.
            mapaEnergii = np.zeros(
                (N_czestosci, book.epoch_s))  # Przygotowujemy tablicę, w której powstanie reprezentacja time-częstość.
            for atom in book.atoms[numerEpoki]:  # iterujemy się po atomach danej epoki
                f_Hz, A, phase, t0, scale = parametry_atomu(book, atom)
                t_mp, atom_time = rekonstrukcja_atomu(book, atom)
                t_mp, f, atom_tfr = tfr_atomu(book, atom, N_czestosci)
                if f_Hz > 11 and f_Hz < 16 and scale >= 0.5:
                    total += 1
                    mapaEnergii += atom_tfr
                    rekonstrukcja += atom_time
                    t_start = t0 + (numerEpoki - 1) * 20 - scale / 2
                    t_end = t0 + (numerEpoki - 1) * 20 + scale / 2
                    mp_spindles.append([t_start, t_end])
            t_mp += (numerEpoki - 1) * 20
            if verbose: TFRPlot(mapaEnergii, t_mp, f, rekonstrukcja, Fs=128, title="{} epoka: {}".format(self.path.split('bin')[0], numerEpoki))

        f_page.table('Start', 'Koniec', np.round(np.array(mp_spindles), 2), 'Przedziały Wrzecion',
                     'Przedziały czasowe [s], w których klasyfikator znalazł wrzeciona snu. Czyli odciniki w których atomy były w przedziale częstości 11Hz - 16Hz i miały szerokość powyżej 0,5s, w sumie znaleziono ' + str(
                         len(mp_spindles)) + ' wrzeciona snu tą metodą.')
        f_page.pagination('./compare.html', './filt.html')
        f_page.close()
        return mp_spindles

if __name__ == '__main__':
    Fs = 128
    signal = Signal('./inb14_fragment.bin', 128)
    t = signal.t
    s = signal.s
    # signal.plot_signal()
    filt_spindles = signal.filt_classifier()
    mp_spindles = signal.mp_classifier(50, 0.1)

    c_page = rg.Page("./report/compare.html", u"Klasyfikator Wrzecion Snu", u"Porównanie Metod", u"Paweł Pierzchlewicz")
    covering = []
    for index, i in enumerate(mp_spindles):
        low = abs(np.array(filt_spindles)[:, 0] - i[0])
        if low[np.where(low == min(low))[0][0]] <= 0.5:
            covering.append([np.where(low == min(low))[0][0], index])  # [filt, mp]
    covering = np.array(covering)
    c_index = 0
    for index, i in enumerate(mp_spindles):
        plt.figure(figsize=((20, 10)))
        if index in covering[:, 1]:
            where = np.where(covering[:, 1] == index)[0][0]
            l = ((np.array(filt_spindles[covering[where][0]]) * Fs).astype('int'),
                 (np.array(mp_spindles[covering[where][1]]) * Fs).astype('int'))
            plt.plot(t[l[0][0] - 50:l[0][1] + 50], s[l[0][0] - 50:l[0][1] + 50])
            plt.axvline(t[l[0][0]], c='r')
            plt.axvline(t[l[0][1]], c='r')
            plt.axvline(t[l[1][0]], c='g')
            plt.axvline(t[l[1][1]], c='g')
            plt.title('Filter & MP')
            plt.savefig("./report/images/comp{}.png".format(i))
            c_index += 1
            c_page.image("./images/comp{}.png".format(i), "Wycinek sygnału {}".format(c_index),
                         u"Wycinek sygnału okienowanego prostokątnie na przedziale {:.2f}s - {:.2f}s.\n Przedstawia wrzeciono znalezione przez MP i poprzez filtrowanie sygnału. Czerwony to przedział znaleziony przez Filtrowanie, a zielony to przedział znaleziony przez MP".format(
                             ((l[0][0] - 50) / Fs), ((l[0][1] + 50)) / Fs))
        else:
            l = (np.array(i) * Fs).astype('int')
            plt.plot(t[l[0] - 50:l[1] + 50], s[l[0] - 50:l[1] + 50])
            plt.axvline(t[l[0]], c='g')
            plt.axvline(t[l[1]], c='g')
            plt.title('MP Only')
            plt.savefig("./report/images/comp{}.png".format(i))
            c_index += 1
            c_page.image("./images/comp{}.png".format(i), "Wycinek sygnału {}".format(c_index),
                     u"Wycinek sygnału okienowanego prostokątnie na przedziale {:.2f}s - {:.2f}s.\n Przedstawia wrzeciono znalezione przez samo MP.".format(
                         (l[0] - 50) / Fs, (l[1] + 50) / Fs))
        # plt.show()
    for index, i in enumerate(filt_spindles):
        if index in covering[:, 1]:
            pass
        else:
            plt.figure(figsize=((20, 10)))
            l = (np.array(i) * Fs).astype('int')
            plt.plot(t[l[0] - 50:l[1] + 50], s[l[0] - 50:l[1] + 50])
            plt.axvline(t[l[0]], c='r')
            plt.axvline(t[l[1]], c='r')
            plt.title('Filter Only')
            plt.savefig("./report/images/comp{}.png".format(i))
            c_index += 1
            c_page.image("./images/comp{}.png".format(i), "Wycinek sygnału {}".format(c_index),
                         u"Wycinek sygnału okienowanego prostokątnie na przedziale {:.2f}s - {:.2f}s.\n Przedstawia wrzeciono znalezione poprzez samo filtrowanie sygnału.".format(
                             int((l[0] - 50) / 128), int((l[1] + 50) / 128)))
            # plt.show()
    c_page.header("Porównanie i Analiza")
    explanation = '''
    Jak widać na podstawie powyższych wykresów metoda filtrowania znalazła znacznie więcej wrzecion. Jest to spowodowane charakterystyką obu metod. Jak wiadomo filtry nie są perfekcyjne i nie odcinamy “schodkowo” częstości, a raczej wykorzystujemy krzywą. Takie filtry wycinają trochę częstotliwości poniżej pożądanej częstości (w przypadku filtrów dolnoprzepustowych) i pozostawiają trochę częstości powyżej. W wyniku tego pozostają struktury, które mają podobną częstotliwość do częstotliwości, w których odcinaliśmy oraz ich moc może być na tyle duża, że przekroczą 5mV. Z tego powodu pojawią się dodatkowe wrzeciona spoza przedziału 11 Hz - 16Hz. Dodatkowo metoda MP jest dużo bardziej bezwzględna i nie pozwala by struktury o częstotliwości poza tego pasma były przepuszczane. Pytanie, czy wrzeciona faktycznie powinny być odcinane tak bezwględnie? Może MP powinno odcinać w trochę szerszym przedziale? Ponadto MP tłumaczy tylko część energii (tu 95%). Część z tych wrzecion, które zostały wykryte przez filtrowanie mogło się zawierać w tych 5% nie wytłumaczonych. W wyniku tych charakterystyk podejrzewamy pojawienie się wrzecion wykrytych tylko przez filtrowanie.
    '''
    c_page.paragraph(explanation)
    explanation = '''
    Pozostają jeszcze te wykryte tylko przez MP. Podobnie jak w poprzednim przypadku, filtr jako nie idealna struktura usuwa także część częstości, które nas interesują. W wyniku tego niektóre wrzeciona mogą zniknąć lub zostać mocno osłabione, przez zostaną pominięte przez 1 metodę.
    '''
    c_page.paragraph(explanation)
    c_page.pagination('./docs.html', './mp.html')
    c_page.close()