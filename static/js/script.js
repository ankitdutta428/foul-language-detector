$(document).ready(function() {
    $('#moderationForm').on('submit', function(e) {
        e.preventDefault();
        const text = $('#text').val();

        $.ajax({
            url: '/moderate',
            type: 'POST',
            data: { text: text },
            success: function(response) {
                // Update the text showing whether the content is appropriate or not
                $('#result').text('Is the content appropriate?: ' + response.moderated);
                
                // Update the percentage bar
                $('#appropriateBar').css('width', response.appropriate_percentage + '%');
                $('#inappropriateBar').css('width', response.inappropriate_percentage + '%');
                
                // Update percentage text
                $('#appropriatePercentage').text(response.appropriate_percentage.toFixed(2) + '%');
                $('#inappropriatePercentage').text(response.inappropriate_percentage.toFixed(2) + '%');
            },
            error: function() {
                $('#result').text('Error occurred while processing.');
            }
        });
    });
});
