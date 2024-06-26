var boxes = $('.whiskycheck');

// If there are no boxes checked, the Compare button is disabled.
boxes.on('change', function() {
    $('#compare').prop('disabled', !boxes.filter(':checked').length);
}).trigger('change');


// If the Select All button is clicked, all checkboxes get checked.
$(function() {
    $(document).on('click', '#checkAll', function() {
        $('.whiskycheck').prop('checked', true);
        $('#compare').prop('disabled', !boxes.filter(':checked').length);
    });    
});


// If the Deselect All button is clicked, all checkboxes get unchecked.
$(function() {
  $(document).on('click', '#uncheckAll', function() {
    $('.whiskycheck').prop('checked', false);
    $('#compare').prop('disabled', !boxes.filter(':checked').length);
  });
});




