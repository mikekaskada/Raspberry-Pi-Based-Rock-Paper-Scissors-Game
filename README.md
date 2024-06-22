# Raspberry-Pi-Based-Rock-Paper-Scissors-Game
[Δείτε το video](https://www.youtube.com/watch?v=iwkqqVQcEus)

## [Έργο 1: Πέτρα, Ψαλίδι, Χαρτί με χρήση Teachable Machine και Raspberry Pi](https://github.com/mikekaskada/Raspberry-Pi-Based-Rock-Paper-Scissors-Game/tree/main/Project_1)

Αυτό το έργο αποτελείται από τρία μέρη:
- **Δημιουργία και επεξεργασία Δεδομένων**, δηλ εικόνων, για το Teachable Machine
- **Εκπαίδευση μοντέλου ΤΝ με Teachable Machine**: Ένα μοντέλο Τεχνητής Νοημοσύνης (ΤΝ) εκπαιδεύτηκε να αναγνωρίζει τις χειρονομίες "Πέτρα, Ψαλίδι, Χαρτί" μέσω της πλατφόρμας Teachable Machine.
- **Παιχνίδι με το Raspberry Pi**: Το Raspberry Pi χρησιμοποιήθηκε για να αναγνωρίζει τις χειρονομίες του παίκτη μέσω της κάμεράς του και να παίζει το παιχνίδι με τυχαίο τρόπο.

## [Έργο 2: Πρόβλεψη Κινήσεων σε "Πέτρα, Ψαλίδι, Χαρτί" με Χρήση Τεχνητής Νοημοσύνης](https://github.com/mikekaskada/Raspberry-Pi-Based-Rock-Paper-Scissors-Game/tree/main/Project_2/Code)

**Για να είναι πιο ενδιαφέρον το παιχνίδι, ο υπολογιστής δεν παίζει συνήθως στην τύχη, αλλά προσπαθεί να ανακαλύψει αν υπάρχουν μοτίβα στον τρόπο με τον οποίο παίζει ο άνθρωπος και προσαρμόζει το παίξιμό του.**
Στο δεύτερο έργο χρησιμοποιήθηκε το εργαλείο Τεχνητής Νοημοσύνης ChatGPT 4ο για να δημιουργηθεί ένας αλγόριθμος που προβλέπει τις επόμενες κινήσεις του παίκτη στο παιχνίδι "Πέτρα, Ψαλίδι, Χαρτί":
- **Ανάλυση Ιστορικών Δεδομένων**: Ο υπολογιστής αναλύει τα ιστορικά δεδομένα από προηγούμενα παιχνίδια για να κατανοήσει τις συνήθειες των παικτών.
- **Συστήματα Μάθησης**: Χρησιμοποιούνται νευρωνικά δίκτυα και μηχανική μάθηση, όπως Δίκτυα Βαθιάς Ενίσχυσης (DQN) και Δίκτυα LSTM για την πρόβλεψη κινήσεων και την ανίχνευση μοτίβων.
- **Προσαρμογή Στρατηγικής**: Ο αλγόριθμος προσαρμόζει τη στρατηγική του ανάλογα με την απόδοσή του και τις κινήσεις των παικτών.

 **Το δεύτερο έργο μπορεί να υλοποιηθεί για λειτουργία τόσο σε Raspberry Pi όσο και σε κοινούς υπολογιστές**, προσφέροντας μια ενδιαφέρουσα εμπειρία παιχνιδιού "Πέτρα, Ψαλίδι, Χαρτί" με χρήση της τεχνολογίας τεχνητής νοημοσύνης.
## [Έργο 3: Αναγώριση χειρονομιών μέσω της κάμερας του Raspberry Pi και χρήση ΤΝ για την στρατηγική με την οποία παίζει ο υπολογιστής](https://github.com/mikekaskada/Raspberry-Pi-Based-Rock-Paper-Scissors-Game/tree/main/Project_3)
Προκειται για τον συνδυασμό των δύο προηγούμενων έργων, δηλαδή της αναγνώρισης των χειρονομιών που κάνει ο παίκτης και το παίξιμο του υπολογιστή με την χρήση ΤΝ με σκοπό να βρεί μοτίβα στο παίξιμο των ανθρώπων.


Μπορείτε να παρακολουθήσετε τα δύο πρώτα έργα [από το video](https://www.youtube.com/watch?v=iwkqqVQcEus)


## Εξοπλισμός
- Raspberry Pi 4 - 4GB
- Raspberry Pi 27W USB-C Power Supply
- Raspberry Pi Case
- microSDHC 32GB Class 10
- HDMI to Micro HDMI Cable
- Raspberry Pi Camera Module V3

## Εξοπλισμός που αγοράστηκε και κόστος
- [Raspberry Pi 4 Model B 4GB Low budget Kit](https://www.hellasdigital.gr/go-create/raspberry-and-accessories/raspberry-pi-4-and-accessories/raspberry-pi-4-model-b-4gb-low-budget-kit-pi4lb4gb/) - €103,39
  ![Raspberry Pi 4 Model B Low budget Kit](https://www.hellasdigital.gr/images/detailed/28/RaspberryPi_4_Model_B_Low_budget_Kit.jpg)
- [Raspberry Pi Camera Module V3](https://www.hellasdigital.gr/go-create/raspberry-and-accessories/accessories/raspberry-pi-camera-module-v3/) - €36,70
  ![Raspberry Pi Camera Module V3](https://www.hellasdigital.gr/images/detailed/32/Standard_Hero__1673252414_444.jpg)

## Συνολικό κόστος: €140,09

### Σημείωση:
Στην πορεία του πρότζεκτ διαπιστώθηκε ότι χρειαζόμασταν μεγαλύτερο καλώδιο για την κάμερα μήκους 1 m το οποίο αγοράστηκε ξεχωριστά.

Επίσης διαπιστώθηακε ότι η κάρτα SD των 32 GΒ ήταν μικρή σε χωρητικότητα.

