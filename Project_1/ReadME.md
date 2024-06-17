## Έργο 1: Πέτρα, Ψαλίδι, Χαρτί με χρήση Teachable Machine και Raspberry Pi

Αυτό το έργο αποτελείται από δύο μέρη:
- **Εκπαίδευση μοντέλου ΤΝ με Teachable Machine**: Ένα μοντέλο Τεχνητής Νοημοσύνης (ΤΝ) εκπαιδεύτηκε να αναγνωρίζει τις χειρονομίες "Πέτρα, Ψαλίδι, Χαρτί" μέσω της πλατφόρμας Teachable Machine.
- **Παιχνίδι με το Raspberry Pi**: Το Raspberry Pi χρησιμοποιήθηκε για να αναγνωρίζει τις χειρονομίες του παίκτη μέσω της κάμεράς του και να παίζει το παιχνίδι με τυχαίο τρόπο.

## Λειτουργικό σύστημα
Μετά από πολλές δυσκολίες και προβλήματα με την εγκατάσταση των απαραίτητων βιβλιοθηκών της Python για την τεχνητή νοημοσύνη, είτε μέσω της εντολής `apt install`, είτε μέσω εικονικού περιβάλλοντος Python (`pip install`) για την βιβλιοθήκη Picamera2, καταλήξαμε στο λειτουργικό **Raspberry Pi 4 Bullseye DNN image**:  _A Raspberry Pi 4 Bullseye 64-OS image with deep learning examples_ από την ομάδα [Q-engineering](https://github.com/Qengineering/RPi-Bullseye-DNN-image).
Λεπτομέρειες για το λειτουργικό σύστημα και τον εξολπισμό υπάρχουν στο αρχείο [Εξοπλισμός.pdf](https://github.com/mikekaskada/Raspberry-Pi-Based-Rock-Paper-Scissors-Game/blob/main/Project_1/%CE%95%CE%BE%CE%BF%CF%80%CE%BB%CE%B9%CF%83%CE%BC%CF%8C%CF%82.pdf).

## Raspberry Pi Camera Module 3
Η [Raspberry Pi Camera Module V3](https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf) είναι μια συμπαγής κάμερα από την Raspberry Pi. Διαθέτει έναν αισθητήρα IMX708  12-megapixel με HDR και υποστηρίζει αυτόματη εστίαση με ανίχνευση φάσης.
Χρειάστηκε να αντικατασταθεί το καλώδιο (cable ribbon) που έχει με άλλο μακρύτερο μήκους 1 m. 

Εκτυπώσαμε με 3D εκτύπωση μια θήκη για την κάμερα [Raspberry Pi Camera Stand for PiCam v3](https://www.thingiverse.com/thing:5805000).

![camera case](https://github.com/mikekaskada/Raspberry-Pi-Based-Rock-Paper-Scissors-Game/blob/main/Project_1/Images/camera%20case.jpg)  

Η κάμερα τοποθετήθηκε πάνω σε στοίβα βιβλίων και κουτιών περίπου 1 m πάνω από την επιφάνεια ενός τραπεζιού. 
Στη βάση της στοίβας τοποθετήσαμε ανεστραμμένο ένα πίνακα και πάνω στον πίνακα ένα σουπλά με την λευκή επιφάνεια προς τα πάνω.

![books](https://github.com/mikekaskada/Raspberry-Pi-Based-Rock-Paper-Scissors-Game/blob/ce576b718da82a72690bebc52997ac0bb25e20ec/Project_1/Images/camera%20on%20top%20of%20books.jpg) ![base](https://github.com/mikekaskada/Raspberry-Pi-Based-Rock-Paper-Scissors-Game/blob/main/Project_1/Images/surface%20with%20white%20background.jpg)  

Με το μονόχρωμο (άσπρο) υπόβαθρο ήταν πιο εύκολη η αναγνώριση και κατηγοριοποίηση της φωτογραφίας αργότερα από το πρόγραμμα.

