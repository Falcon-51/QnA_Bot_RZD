function openReg() {
	var modal = document.getElementById("formReg");
	modal.style.display = "block";
}

function closeReg() {
	var modal = document.getElementById("formReg");
	modal.style.display = "none";
}

function openAuth() {
	var modal = document.getElementById("formAuth");
	modal.style.display = "block";
}

function closeAuth() {
	var modal = document.getElementById("formAuth");
	modal.style.display = "none";
}

function addData(event) {
    event.preventDefault(); 
    const email = $('#mail').val();
    const login = $('#login').val();
    const password = $('#password').val();
    $.ajax({
        url: 'http://localhost:8000/register',  
        type: 'POST',
        contentType: 'application/json',  
        data: JSON.stringify({ email: email, login: login, password: password }),  
        success: function(response) {
            console.log('Успех:', response);  
            alert('Регистрация успешна!');
            closeReg();  
        },
        error: function(xhr, status, error) {
            console.error('Ошибка:', error);  
            alert('Произошла ошибка при регистрации.');
        }
    });
}

function verifyData(event) {
    event.preventDefault(); 
    const login = $('#login').val();
    const password = $('#password').val();
    $.ajax({
        url: 'http://localhost:8000/auth',  
        type: 'POST',
        contentType: 'application/json',  
        data: JSON.stringify({ login: login, password: password }),  
        success: function(response) {
            console.log('Успех:', response);  
            alert('Авторизация успешна!');
            localStorage.setItem('login', login);
            closeReg();  
            updateUI();
        },
        error: function(xhr, status, error) {
            console.error('Ошибка:', error);  
            alert('Произошла ошибка при регистрации.');
        }
    });
}

function updateUI() {
    const login = localStorage.getItem('login');
    if (login) {
        document.getElementById('login').style.display = 'none';
        document.getElementById('reg').style.display = 'none';
        document.getElementById('chat').style.display = 'block';
        document.getElementById('document').style.display = 'block';
        const greetingMessage = document.getElementById('greetingMessage');
        document.getElementById('greetingText').textContent = `Привет, ${login}`;
        greetingMessage.style.display = 'block';
    }
}

const sendButton = document.getElementById('send-btn');
const userInput = document.getElementById('user-input');
const chatBox = document.getElementById('chat-box');

function appendMessage(message, isUser = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser ? 'text-end' : 'text-start';
    messageDiv.innerHTML = `<p>${message}</p>`;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
    const userMessage = userInput.value.trim();
    if (userMessage === "") 
        return;

    // Добавляем сообщение пользователя в чат
    appendMessage(userMessage);

    // Очищаем поле ввода
    userInput.value = '';

    // Добавляем сообщение о процессе обработки
    appendMessage('Пожалуйста, подождите...', false);

    // Создаём объект для отправки на сервер
    const data = { message: userMessage };

    // Отправляем запрос на сервер (заменить URL на ваш сервер)
    fetch('https://example.com/api/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
    // Удаляем сообщение "Пожалуйста, подождите..."
    chatBox.lastChild.remove();

    // Отображаем ответ сервера
    appendMessage(data.response, false);
    })
    .catch(error => {
    console.error('Error:', error);
    chatBox.lastChild.remove();
    appendMessage('Произошла ошибка. Попробуйте снова.', false);
    });
    }

    sendButton.addEventListener('click', sendMessage);

    // Отправка сообщения по нажатию Enter
    userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
    sendMessage();
    }
});

// addData() {
//     // Функция для добавление записи
// }
// deleteData() {
//     // Функция для удаления записи
// }