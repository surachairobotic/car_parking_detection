import pymysql
import random
from datetime import datetime

class CarParkingDatabase:
    def __init__(self, host, user, password, port, database):
        # Initialize the connection to the MySQL database
        self.connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            port=port,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    
    def insert_gate_event(self, media_link):
        try:
            with self.connection.cursor() as cursor:
                # Generate random values for gate, event, and vehicle_type
                gate = random.choice(['A', 'B', 'C'])
                event = random.choice(['IN', 'OUT'])
                vehicle_type = random.choice(['CAR', 'TRUCK'])

                # Get the current server_time and timestamp
                server_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # SQL query to insert data into gate_events table
                sql = """
                    INSERT INTO gate_events (server_time, gate, event, vehicle_type, media_link, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                # Execute the SQL query
                cursor.execute(sql, (server_time, gate, event, vehicle_type, media_link, timestamp))
            
            # Commit the transaction
            self.connection.commit()
            print("Data inserted successfully.")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            self.connection.rollback()
    
    def close(self):
        # Close the database connection
        self.connection.close()

# Example usage:
# db = CarParkingDatabase(host='localhost', user='root', password='your_password', database='car_parking')
# db.insert_gate_event(media_link='http://example.com/image.jpg')
# db.close()
