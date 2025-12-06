import pymysql

try:
    # Connexion à MySQL
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',  # Vide par défaut dans XAMPP
        database='bikepred_prod',
        charset='utf8mb4'
    )
    
    print("✅ Connexion MySQL réussie!")
    
    # Test simple
    with connection.cursor() as cursor:
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"📊 Version MySQL: {version[0]}")
    
    connection.close()
    print("✅ Test terminé avec succès")
    
except pymysql.Error as e:
    print(f"❌ Erreur de connexion: {e}")