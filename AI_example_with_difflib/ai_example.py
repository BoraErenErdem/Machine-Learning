import csv
import os
from difflib import get_close_matches as yakin_sonuclari_getir

# Veritabanı dosya yolu
veritabani_dosyasi = 'C:/Users/Boranın pc/PycharmProjects/Machine Learning/AI_example_with_difflib/Veritabani.csv'


def veritabanini_yukle():
    veriler = []
    if os.path.exists(veritabani_dosyasi):
        with open(veritabani_dosyasi, mode='r', encoding='utf-8') as dosya:
            csv_reader = csv.DictReader(dosya)
            for satir in csv_reader:
                veriler.append({'soru': satir['soru'], "cevap": satir['cevap']})
    return veriler


def veritabanina_yaz(veriler):
    with open(veritabani_dosyasi, mode='w', newline='', encoding='utf-8') as dosya:
        fieldnames = ['soru', 'cevap']
        csv_writer = csv.DictWriter(dosya, fieldnames=fieldnames)
        csv_writer.writeheader()
        for veri in veriler:
            csv_writer.writerow(veri)


def yakin_sonuc_bul(soru, sorular):
    eslesen = yakin_sonuclari_getir(soru, sorular, n=1, cutoff=0.6)
    return eslesen[0] if eslesen else None


def cevabini_bul(soru, veritabani):
    for soru_cevaplar in veritabani:
        if soru_cevaplar["soru"] == soru:
            return soru_cevaplar["cevap"]
    return None


def chat_bot():
    veritabani = veritabanini_yukle()

    while True:
        soru = input("Siz: ")
        if soru.lower() == 'çık':
            break

        gelen_sonuc = yakin_sonuc_bul(soru, [soru_cevaplar["soru"] for soru_cevaplar in veritabani])
        if gelen_sonuc:
            verilecek_cevap = cevabini_bul(gelen_sonuc, veritabani)
            print(f"Bot: {verilecek_cevap}")
        else:
            print("Bot: Bunu nasıl cevaplayacağımı bilmiyorum. Öğretir misiniz?")
            yeni_cevap = input("Öğretmek için yazabilir veya 'geç' diyebilirsiniz. ")

            if yeni_cevap != 'geç':
                veritabani.append({
                    "soru": soru,
                    "cevap": yeni_cevap
                })
                veritabanina_yaz(veritabani)
                print("Bot: Teşekkürler, sayenizde yeni bir şey öğrendim.")


chat_bot()