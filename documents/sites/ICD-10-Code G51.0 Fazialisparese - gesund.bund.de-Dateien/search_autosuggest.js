/* eslint-disable */

const CONFIGURE_AUTO_SUGGEST = () => {
    const SECRET_KEY = '4C4FZZ646FCB938458E66249C75B8';
    const URL_SEARCH_AUTOCOMPLETE = '/service.api.search.autocomplete';

    $('.search-form__input.typeahead').each(function () {
        $(this).typeahead({
            autoSelect: false,
            items: 10,
            minLength: 3,
            selectOnBlur: false,
            highlighter: function (item) {
                var text = this.query;
                if (text === '') {
                    return item;
                }
                var matches = item.match(/(>)([^<]*)(<)/g);
                var first = [];
                var second = [];
                var i;
                if (matches && matches.length) {
                    // html
                    for (i = 0; i < matches.length; ++i) {
                        if (matches[i].length > 2) {// escape '><'
                            first.push(matches[i]);
                        }
                    }
                } else {
                    // text
                    first = [];
                    first.push(item);
                }
                text = text.replace((/[\(\)\/\.\*\+\?\[\]]/g), function (mat) {
                    return '\\' + mat;
                });
                var reg = new RegExp(text, 'gi');
                var m;
                for (i = 0; i < first.length; ++i) {
                    m = first[i].match(reg);
                    if (m && m.length > 0) {// find all text nodes matches
                        second.push(first[i]);
                    }
                }
                for (i = 0; i < second.length; ++i) {
                    item = item.replace(second[i], second[i].replace(reg, '<strong>$&</strong>'));
                }

                return item;
            },
            afterSelect(item) {
                this.$element[0].form.submit();
            },
            updater(item) {
                this.$menu[0].querySelector('.active').scrollIntoView(false);
                return item;
            },
            source(query, process) {
                return $.post({
                    url: URL_SEARCH_AUTOCOMPLETE,
                    data: JSON.stringify({ q: query, secret: SECRET_KEY, lang: document.documentElement.lang }),
                    success: (data) => process(data.data)
                });
            },
        })
        .on('keydown', function(e) {   
            if (e.key === 'Enter') {
                console.log('submit current value');
                this.form.submit();
            }
        })
    });

    $(".search-form input").focus(function () {
        $(this).closest('form').find('.search-form__submit').addClass("isFocus");
    }), $(".search-form input").blur(function () {
        "" == $(this).closest('form').find('.search-form__submit').val() && $(this).closest('form').find('.search-form__submit').removeClass("isFocus");
    });
};

CONFIGURE_AUTO_SUGGEST();
