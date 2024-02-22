import rospy
from gs_flight import FlightController, CallbackEvent
from gs_board import BoardManager
from gs_module import ModuleLedController

rospy.init_node("flight_test_node") # инициализируем ноду в системе ROS
coordinates = [
    # [1.4,3.5,0.7],
    [2.7,4.6,1.7],
    [0.2,4.6,1.7],
    [0.2,2.3,1.7],
    [0.4,0.2,1.7],
    [2.6,0.25,1.7],
    [2.6,2.3,1.7],
    [1.3,1.2,0.85]] # создаем массив точек, по которым будет лететь

run = True # переменная отвечающая за работу программы, значение True программа

position_number = 0 # счетчик пройденных точек

moduleLed = ModuleLedController()
def callback(event): # функция обработки событий Автопилота
 global ap
 global run
 global coordinates
 global position_number

 event = event.data # преобразуем сообщение ROS std_msgs/Int32 в Питоновский int
# если событие Автопилота ENGINES_STARTED( моторы заведены), то выполняем взлет

 # иначе делаем проверку на другое событие Автопилота
 if event == CallbackEvent.ENGINES_STARTED:
    print("engine started") # выводим на экран сообщение, что двигатели

    ap.takeoff() # отдаем команду на взлет, как только она будет выполнена
 

# иначе делаем проверку на другое событие Автопилота
 elif event == CallbackEvent.TAKEOFF_COMPLETE:
    moduleLed.changeAllColor(0.0, 255.0, 0.0) 
    rospy.sleep(1) 
    moduleLed.changeAllColor(0.0,0.0,0.0)
    print("takeoff complite") # выводим на экран сообщение, что взлет закончен
    position_number = 0 # обнуляем счетчик точек
 # отдаем Автопилоту команду на перемещение в нулевую точку,
 # как только команда завершится
 # Автопилот сгенерирует POINT_REACHED(точка достигнута)
    ap.goToLocalPoint(coordinates[position_number][0],
    coordinates[position_number][1],
    coordinates[position_number][2])
# если событие Автопилота POINT_REACHED (точка достигнута),
 # то отдаем команду на перемещение в следующую точку или
 # отдаем команду на приземление
    rospy.sleep(1)
 elif event == CallbackEvent.POINT_REACHED:
    moduleLed.changeAllColor(0.0, 0.0, 255.0) 
    rospy.sleep(0.4) 
    moduleLed.changeAllColor(0.0,0.0,0.0)
    print("point {} reached".format(position_number)) # выводим на экран номер

    position_number += 1 # наращиваем счетчик пройденных точек на 1
    # делаем проверку: если количество пройденных точек меньше чем количество

 # в массиве со всеми точками, то выполняем перемещение в следующую точку,
# иначе делаем проверку: если количество пройденных точек равно количеству

# в массиве со всеми точками, то выполняем перемещение в точку ¾Дом¿,
# иначе выполняем посадку
    if position_number < len(coordinates):
# перемещение в следующую точку
 # как только команда завершится
 # Автопилот сгенерирует POINT_REACHED(точка достигнута)
        ap.goToLocalPoint(coordinates[position_number][0],
        coordinates[position_number][1],
        coordinates[position_number][2])

#     elif position_number == len(coordinates):
#  # перемещение в точку ¾Дом¿
#  # как только команда завершится
#  # Автопилот сгенерирует POINT_REACHED(точка достигнута)
#         ap.goToLocalPoint(coordinates[position_number][0],
#         coordinates[position_number][1],
#         coordinates[position_number][2])
    else:
        ap.disarm()
        # если событие Автопилота равно COPTER_LANDED (совершена посадка),
# то прекращаем выполнение программы
 elif event == CallbackEvent.COPTER_LANDED:
    print("finish programm")
 # устанавливаем переменную,
# отвечающую за работу программы в значение False - программа должна

    run = False

 # создаем объект класса получения бортовой информации
board = BoardManager()
 # создаем объект управления полета Автопилотом, передаем функцию обработки
ap = FlightController(callback)

 # переменная, отвечающая за однократное выполнения блока предстартовой подготовки.
once = False

 # цикл во время выполнение которого работает функция callback
 # Этот цикл работает пока переменная run в значении True и пока ROS включен
while not rospy.is_shutdown() and run:
 # если плата Автопилота подключена к Raspberry и переменная once равна False,
 # то выполняем предстартовую подготовку
    if board.runStatus() and not once:
        print("start programm")
 # отдает команду Автопилоту выполнить предстартовую подготовку,
 # как только она будет выполнена
 # Автопилот сгенерирует событие ENGINES_STARTED (Двигатели заведены)
        ap.preflight()
 # устанавливаем в True, чтобы больше не выполнять блок предстартовой

        once = True
    pass