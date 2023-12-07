css = '''
<style>
.chat-container {
    border: 2px solid #dedede;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
.chat-container::after {
    content: "";
    clear: both;
    display: table;
}
.chat-container.darker {
    border-color: #ccc;
}
.chat-avatar {
    width: 100%;
    padding-bottom: 10px;
}
.chat-message {
    width: 100%;
    padding-bottom: 10px;
}
.chat-message p {
    padding: 5px 10px;
    border-radius: 10px;
    display: inline-block;
}
.chat-message p img {
    max-width: 100%;
    height: auto;
}
.chat-message p::after {
    content: "";
    clear: both;
    display: table;
}
</style>
'''

bot_template = '''
<div class="chat-container">
    <div class="chat-message">
        <p>{}</p>
    </div>
</div>
'''

user_template = '''
<div class="chat-container darker">
    <div class="chat-message">
        <p>{}</p>
    </div>
</div>
'''